"""
STEP 4 — Shapelet analysis.

A **shapelet** is a short subsequence that tends to appear in one class and not
the other. Unlike a whole-series classifier, a shapelet gives a rule you can
read out loud: *"if this 40-step pattern appears anywhere in a child's
activity trace, they fall on this side of the split."*

The stage does four things:

1. Measure how far a shapelet sits from every training subject
   (:func:`subsequence_dist_znorm` and its compiled twin).
2. Turn that distance list into a discriminative score by finding the split
   point that maximises information gain (:func:`find_best_split`).
3. Do it faster with two classic optimisations — early abandoning and
   admissible entropy pruning (:func:`evaluate_shapelets`).
4. Read the winner back as a plain-English rule (:func:`extract_rule`).

**On distances.** Every comparison here is Z-normalised: both the shapelet and
each sliding window are shifted to mean 0 and scaled to standard deviation 1
before they are compared. That makes matching invariant to amplitude and
baseline, so it responds to the *shape* of a pattern rather than how strong the
signal happened to be.

**On operation counts.** The brute-force baseline reported in the audit is
computed analytically — ``n_subjects x (n_timesteps - L + 1) x L`` — rather
than by running a second, slower pass over the data.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

from src.config import (CLASS_NAMES_LONG, N_TIMESTEPS, PATHS,
                        PRUNING_CHECK_FRACTION, PRUNING_GUARD_FRACTION,
                        RANDOM_SHAPELET_COUNT, RANDOM_SHAPELET_LENGTHS,
                        SEEDS, SHAPELET_CANDIDATES, ZNORM_EPSILON)
from src.utils import z_normalize


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def extract_candidate_shapelets(X_train, y_train):
    """Cut the hand-picked candidate subsequences out of the training data.

    Each candidate is stored twice: the raw values (for plotting a recognisable
    shape) and the Z-normalised values (what the distance functions consume).

    Args:
        X_train: Training array of shape ``(n_subjects, 2, n_timesteps)``.
        y_train: Training labels.

    Returns:
        List of dicts describing each candidate shapelet.
    """
    shapelets = []
    for candidate_id, subject, channel, start, length in SHAPELET_CANDIDATES:
        values = X_train[subject, channel, start:start + length]
        shapelets.append({
            "id": candidate_id,
            "subject": subject,
            "channel": channel,
            "start": start,
            "length": length,
            "label": y_train[subject],
            "values": values,
            "values_norm": z_normalize(values, ZNORM_EPSILON),
        })

    print(f"Extracted {len(shapelets)} candidate shapelets from training "
          "data.")
    return shapelets


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def subsequence_dist_euclidean(series, shapelet):
    """Slide a shapelet along a series and return its closest raw-value match.

    This is the textbook definition without normalisation, kept because it is
    the clearest statement of what "subsequence distance" means: try every
    alignment, keep the best. Because it skips normalisation it still responds
    to amplitude, which is why the analysis below uses the Z-normalised version
    instead.

    Args:
        series: 1-D target series.
        shapelet: 1-D shapelet, raw (un-normalised) values.

    Returns:
        Tuple ``(minimum_distance, best_start_index)``.
    """
    length = len(shapelet)
    min_distance = np.inf
    best_index = 0

    for start in range(len(series) - length + 1):
        distance = np.sqrt(np.sum((series[start:start + length] - shapelet)**2))
        if distance < min_distance:
            min_distance = distance
            best_index = start

    return min_distance, best_index


def subsequence_dist_znorm(series, shapelet_norm):
    """Same sliding search, but Z-normalising each window before comparing.

    This is the metric every result in this stage is based on. Normalising the
    window as well as the shapelet is what buys amplitude invariance: two
    stretches with the same shape score as identical however large the swing.

    Args:
        series: 1-D target series.
        shapelet_norm: 1-D Z-normalised shapelet.

    Returns:
        Tuple ``(minimum_distance, best_start_index)``.
    """
    length = len(shapelet_norm)
    min_distance = np.inf
    best_index = 0

    for start in range(len(series) - length + 1):
        window = z_normalize(series[start:start + length], ZNORM_EPSILON)
        distance = np.sqrt(np.sum((window - shapelet_norm)**2))
        if distance < min_distance:
            min_distance = distance
            best_index = start

    return min_distance, best_index


@jit(nopython=True)
def subsequence_dist_early_abandon(series, shapelet_norm):
    """Compiled Z-normalised subsequence distance with early abandoning.

    Two speedups over :func:`subsequence_dist_znorm`, with an identical result:

    * **Prefix sums** give each window's mean and standard deviation in O(1)
      instead of O(L), so normalising every window costs nothing extra.
    * **Early abandoning** stops accumulating a window's squared distance the
      moment it exceeds the best distance found so far in this series — that
      window cannot win, so the remaining terms are wasted work.

    Args:
        series: 1-D target series.
        shapelet_norm: 1-D Z-normalised shapelet.

    Returns:
        Tuple ``(minimum_distance, operations_performed)``, where the second
        value counts element comparisons for the speedup audit.
    """
    length = len(shapelet_norm)
    n_steps = len(series)
    min_distance_squared = np.inf
    operations = 0

    # Running sums of the series and of its squares, so any window's mean and
    # variance can be read off in constant time.
    cumulative = np.zeros(n_steps + 1)
    cumulative_squares = np.zeros(n_steps + 1)
    for index in range(n_steps):
        cumulative[index + 1] = cumulative[index] + series[index]
        cumulative_squares[index + 1] = (cumulative_squares[index]
                                         + series[index] * series[index])

    for start in range(n_steps - length + 1):
        window_sum = cumulative[start + length] - cumulative[start]
        window_sum_squares = (cumulative_squares[start + length]
                              - cumulative_squares[start])
        mean = window_sum / length
        variance = window_sum_squares / length - mean * mean
        deviation = np.sqrt(variance) if variance > 0.0 else 0.0
        deviation += ZNORM_EPSILON

        distance_squared = 0.0
        for offset in range(length):
            normalized = (series[start + offset] - mean) / deviation
            difference = normalized - shapelet_norm[offset]
            distance_squared += difference * difference
            operations += 1
            # Early abandon: this window is already worse than the best match.
            if distance_squared >= min_distance_squared:
                break
        else:
            if distance_squared < min_distance_squared:
                min_distance_squared = distance_squared

    return np.sqrt(min_distance_squared), operations


# ---------------------------------------------------------------------------
# Information gain
# ---------------------------------------------------------------------------

def entropy_bin(p_class0, p_class1):
    """Binary entropy, in bits, evaluated element-wise.

    Entropy measures label impurity: 1.0 bit when a group is an even mix of
    both classes, 0.0 when it is pure.

    Probabilities are clipped away from exactly 0 and 1 first. If floating-point
    drift ever pushed one fractionally outside the valid range, ``log2`` would
    blow up; clipping makes the function safe for both the scalar (baseline)
    and vector (split-search) call sites without changing the normal result.

    Args:
        p_class0: Array of class-0 proportions.
        p_class1: Array of class-1 proportions.

    Returns:
        Array of entropies, same shape as the inputs.
    """
    p_class0 = np.clip(p_class0, 1e-15, 1.0 - 1e-15)
    p_class1 = np.clip(p_class1, 1e-15, 1.0 - 1e-15)

    entropy = np.zeros_like(p_class0, dtype=float)
    positive0 = p_class0 > 0
    entropy[positive0] -= p_class0[positive0] * np.log2(p_class0[positive0])
    positive1 = p_class1 > 0
    entropy[positive1] -= p_class1[positive1] * np.log2(p_class1[positive1])
    return entropy


def find_best_split(distances, labels):
    """Find the distance threshold that best separates the two classes.

    Sort the subjects by their distance to the shapelet, then consider every
    boundary between two neighbouring distances. Each boundary splits the data
    in two; the best one is whichever reduces label entropy the most.

    The whole sweep is vectorised with cumulative sums, so all N-1 candidate
    boundaries are scored in one pass rather than one loop iteration each.

    Args:
        distances: Distance from the shapelet to each subject.
        labels: Binary label of each subject, aligned with ``distances``.

    Returns:
        Tuple ``(best_gain, best_split_point)``. Both are 0.0 when every
        distance is identical, since no boundary exists.
    """
    n_subjects = len(distances)
    order = np.argsort(distances)
    distances_sorted = distances[order]
    labels_sorted = labels[order]

    # Entropy of the whole set, before any split.
    total_class0 = np.sum(labels == 0)
    total_class1 = np.sum(labels == 1)
    base_entropy = entropy_bin(np.array([total_class0 / n_subjects]),
                               np.array([total_class1 / n_subjects]))[0]

    # Running class counts, so the composition of both sides of any boundary
    # can be read off directly.
    cumulative_class1 = np.cumsum(labels_sorted)
    cumulative_class0 = np.arange(1, n_subjects + 1) - cumulative_class1

    left_size = np.arange(1, n_subjects)
    left_class1 = cumulative_class1[:-1]
    left_class0 = cumulative_class0[:-1]
    right_class1 = total_class1 - left_class1
    right_class0 = total_class0 - left_class0
    right_size = n_subjects - left_size

    entropy_left = entropy_bin(left_class0 / left_size,
                               left_class1 / left_size)
    entropy_right = entropy_bin(right_class0 / right_size,
                                right_class1 / right_size)

    # Information gain = impurity removed by splitting here.
    weighted_entropy = ((left_size / n_subjects) * entropy_left
                        + (right_size / n_subjects) * entropy_right)
    gains = base_entropy - weighted_entropy

    # A boundary is only meaningful between two *distinct* distances.
    is_valid = distances_sorted[:-1] < distances_sorted[1:]
    if not np.any(is_valid):
        return 0.0, 0.0

    valid_positions = np.where(is_valid)[0]
    best_valid = np.argmax(gains[valid_positions])
    best_position = valid_positions[best_valid]

    best_gain = gains[valid_positions][best_valid]
    # Put the threshold halfway between the two distances it separates.
    best_split = (distances_sorted[best_position]
                  + distances_sorted[best_position + 1]) / 2.0
    return best_gain, best_split


def calculate_entropy_upper_bound(seen_distances, seen_labels, total_class0,
                                  total_class1):
    """Best information gain this candidate could still reach, at best.

    Used to abandon hopeless candidates early. Having scored only part of the
    training set, we ask: if every remaining subject fell in the most flattering
    possible place, how high could the gain get? Because entropy is convex, that
    optimum always sits at one of four corners — all remaining subjects to the
    left, all to the right, or the two classes split cleanly between the sides.

    If even that optimistic ceiling is below the best gain already achieved by
    another shapelet, this candidate cannot win and the scan stops.

    Args:
        seen_distances: Distances computed so far.
        seen_labels: Labels of the subjects scored so far.
        total_class0: Total class-0 count in the full training set.
        total_class1: Total class-1 count in the full training set.

    Returns:
        The upper bound on achievable information gain.
    """
    n_total = total_class0 + total_class1
    n_seen = len(seen_distances)

    remaining_class0 = total_class0 - np.sum(seen_labels == 0)
    remaining_class1 = total_class1 - np.sum(seen_labels == 1)

    base_entropy = entropy_bin(np.array([total_class0 / n_total]),
                               np.array([total_class1 / n_total]))[0]

    order = np.argsort(seen_distances)
    distances_sorted = seen_distances[order]
    labels_sorted = seen_labels[order]

    cumulative_class1 = np.cumsum(labels_sorted)
    cumulative_class0 = np.arange(1, n_seen + 1) - cumulative_class1
    best_bound = 0.0

    for position in range(n_seen - 1):
        if distances_sorted[position] == distances_sorted[position + 1]:
            continue

        seen_left_class1 = cumulative_class1[position]
        seen_left_class0 = cumulative_class0[position]
        seen_right_class1 = cumulative_class1[-1] - seen_left_class1
        seen_right_class0 = cumulative_class0[-1] - seen_left_class0

        # The four extreme placements of the not-yet-scored subjects.
        corners = [
            (remaining_class0, remaining_class1),  # all remaining go left
            (remaining_class0, 0),                 # class 0 left, class 1 right
            (0, remaining_class1),                 # class 1 left, class 0 right
            (0, 0),                                # all remaining go right
        ]

        for extra_class0, extra_class1 in corners:
            left_class0 = seen_left_class0 + extra_class0
            left_class1 = seen_left_class1 + extra_class1
            right_class0 = (seen_right_class0
                            + (remaining_class0 - extra_class0))
            right_class1 = (seen_right_class1
                            + (remaining_class1 - extra_class1))

            left_size = left_class0 + left_class1
            right_size = right_class0 + right_class1
            if left_size == 0 or right_size == 0:
                continue

            entropy_left = entropy_bin(np.array([left_class0 / left_size]),
                                       np.array([left_class1 / left_size]))[0]
            entropy_right = entropy_bin(
                np.array([right_class0 / right_size]),
                np.array([right_class1 / right_size]))[0]

            weighted_entropy = ((left_size / n_total) * entropy_left
                                + (right_size / n_total) * entropy_right)
            best_bound = max(best_bound, base_entropy - weighted_entropy)

    return best_bound


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_shapelets(shapelets, X_train, y_train):
    """Score every candidate shapelet, pruning the hopeless ones early.

    For each candidate the training subjects are scanned in a fixed shuffled
    order (seeded, so the run is reproducible). Shuffling matters: scanning in
    dataset order would let a run of same-class subjects at the start make the
    entropy bound wildly optimistic or pessimistic.

    Args:
        shapelets: Candidates from :func:`extract_candidate_shapelets`.
        X_train: Training array ``(n_subjects, 2, n_timesteps)``.
        y_train: Training labels.

    Returns:
        List of result dicts, one per candidate, in candidate order.
    """
    results = []
    best_gain_so_far = 0.0
    n_subjects = len(X_train)
    total_class0 = np.sum(y_train == 0)
    total_class1 = np.sum(y_train == 1)

    scan_order = np.random.default_rng(
        seed=SEEDS["shapelet_order"]).permutation(n_subjects)

    # Pruning is only considered after the guard window, then re-checked at
    # regular intervals rather than after every single subject.
    guard_size = int(PRUNING_GUARD_FRACTION * n_subjects)
    check_interval = int(PRUNING_CHECK_FRACTION * n_subjects)

    print(f"Evaluating {len(shapelets)} candidate shapelets...")
    for shapelet in shapelets:
        shapelet_norm = shapelet["values_norm"]
        channel = shapelet["channel"]
        length = shapelet["length"]

        operations = 0
        seen_distances = []
        seen_labels = []
        was_pruned = False
        samples_evaluated = 0
        start_time = time.time()

        for position, subject_index in enumerate(scan_order):
            series = X_train[subject_index, channel]

            distance, subject_operations = subsequence_dist_early_abandon(
                series, shapelet_norm)
            operations += subject_operations
            seen_distances.append(distance)
            seen_labels.append(y_train[subject_index])

            scanned = position + 1
            if scanned >= guard_size and scanned % check_interval == 0:
                upper_bound = calculate_entropy_upper_bound(
                    np.array(seen_distances), np.array(seen_labels),
                    total_class0, total_class1)
                if upper_bound < best_gain_so_far:
                    was_pruned = True
                    samples_evaluated = scanned
                    break

        # Put the distances back in original subject order so they line up with
        # y_train for the split search.
        distances = np.zeros(n_subjects)
        for position, value in enumerate(seen_distances):
            distances[scan_order[position]] = value

        if was_pruned:
            # A pruned candidate is never scored: it already cannot win.
            gain, split_point = 0.0, 0.0
        else:
            samples_evaluated = n_subjects
            gain, split_point = find_best_split(distances, y_train)
            best_gain_so_far = max(best_gain_so_far, gain)

        results.append({
            "id": shapelet["id"],
            "channel": channel,
            "length": length,
            "gain": gain,
            "split_point": split_point,
            "pruned": was_pruned,
            "pruned_at_sample": samples_evaluated,
            # Brute force would compare every window element of every subject.
            "ops_std": n_subjects * (N_TIMESTEPS - length + 1) * length,
            "ops_ea": operations,
            "time": time.time() - start_time,
            "distances": distances,
        })

        print(f"  Shapelet {shapelet['id']:2d} | Channel {channel} | "
              f"Length {length:2d} | Gain: {gain:.6f} | "
              f"Split: {split_point:.4f} | Pruned: {str(was_pruned):5s} "
              f"(at {samples_evaluated})")

    return results


def print_leaderboard(results):
    """Rank the candidates by information gain and report the speedup audit.

    Args:
        results: Output of :func:`evaluate_shapelets`.

    Returns:
        The results sorted by gain, best first.
    """
    ranked = sorted(results, key=lambda row: row["gain"], reverse=True)

    print("\n=== RANKED LEADERBOARD OF SHAPELETS ===")
    print(f"{'Rank':>4} | {'ID':>4} | {'Info Gain':>10} | "
          f"{'Split Thresh':>12} | {'Channel':>7} | {'Length':>6}")
    print("-" * 58)
    for rank, row in enumerate(ranked):
        print(f"{rank + 1:4d} | {row['id']:4d} | {row['gain']:10.6f} | "
              f"{row['split_point']:12.4f} | {row['channel']:7d} | "
              f"{row['length']:6d}")

    # How much work the two optimisations actually saved.
    total_brute_force = sum(row["ops_std"] for row in results)
    total_early_abandon = sum(row["ops_ea"] for row in results)
    saved = total_brute_force - total_early_abandon

    print("\n=== SPEEDUP AUDIT STATISTICS ===")
    print(f"  Standard Brute-Force Operations:    {total_brute_force:,}")
    print(f"  Distance Early Abandon Operations:  {total_early_abandon:,}")
    print(f"  Operations Saved:                   {saved:,} "
          f"({saved / total_brute_force * 100:.2f}% savings)")

    pruned_count = sum(1 for row in results if row["pruned"])
    print(f"  Admissible Entropy Pruning: Pruned {pruned_count} out of "
          f"{len(results)} candidates early.")

    return ranked


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def compare_with_random_shapelets(X_train, y_train, best_gain):
    """Score K randomly drawn subsequences as a sanity check on the candidates.

    If a handful of random cuts match a deliberately chosen pool, the pool is
    not actually carrying information — an important negative control, and the
    reason this comparison uses exactly the same Z-normalised metric.

    Args:
        X_train: Training array ``(n_subjects, 2, n_timesteps)``.
        y_train: Training labels.
        best_gain: Best gain from the hand-picked pool, for the printout.

    Returns:
        List of information gains, one per random shapelet.
    """
    rng = np.random.default_rng(seed=SEEDS["random_shapelet"])
    random_gains = []

    for _ in range(RANDOM_SHAPELET_COUNT):
        subject = rng.integers(0, len(X_train))
        channel = rng.integers(0, 2)
        length = rng.choice(RANDOM_SHAPELET_LENGTHS)
        start = rng.integers(0, N_TIMESTEPS - length)

        shapelet_norm = z_normalize(
            X_train[subject, channel, start:start + length], ZNORM_EPSILON)

        distances = np.array([
            subsequence_dist_znorm(X_train[index, channel], shapelet_norm)[0]
            for index in range(len(X_train))
        ])
        gain, _ = find_best_split(distances, y_train)
        random_gains.append(gain)

    print(f"=== Random Shapelets (K={RANDOM_SHAPELET_COUNT}) vs Discrete "
          "Pool ===")
    print(f"  Random Shapelets Max Gain:  {np.max(random_gains):.6f}")
    print(f"  Random Shapelets Mean Gain: {np.mean(random_gains):.6f}")
    print(f"  Our Pool Best Shapelet Gain: {best_gain:.6f}")
    return random_gains


# ---------------------------------------------------------------------------
# Rule extraction and visualisation
# ---------------------------------------------------------------------------

def extract_rule(ranked_results, shapelets):
    """Turn the top-ranked shapelet into a readable if/else classification rule.

    Subjects closer to the shapelet than its threshold are assigned the class
    of the subject the shapelet was cut from; everyone else gets the other
    class.

    Args:
        ranked_results: Sorted results from :func:`print_leaderboard`.
        shapelets: Candidate list from :func:`extract_candidate_shapelets`.

    Returns:
        Tuple ``(best_result, best_shapelet)`` for the winning candidate.
    """
    best_result = ranked_results[0]
    best_shapelet = next(candidate for candidate in shapelets
                         if candidate["id"] == best_result["id"])

    # The shapelet votes for whichever class its source subject belongs to.
    if best_shapelet["label"] == 0:
        near_class, far_class = CLASS_NAMES_LONG[0], CLASS_NAMES_LONG[1]
    else:
        near_class, far_class = CLASS_NAMES_LONG[1], CLASS_NAMES_LONG[0]

    print("=== Factual Classification Rule ===")
    print(f"If Time Series Q contains a subsequence within a distance "
          f"threshold of {best_result['split_point']:.4f} matching Shapelet "
          f"{best_result['id']}, then classify as {near_class}; "
          f"else {far_class}.")

    return best_result, best_shapelet


def plot_shapelet_alignment(best_result, best_shapelet, X_train, y_train):
    """Show where the winning shapelet matches in four example subjects.

    Two subjects from each class, so the visual difference between a close
    match and a poor one is directly comparable.

    Args:
        best_result: Winning row from the leaderboard.
        best_shapelet: The matching candidate dict.
        X_train: Training array ``(n_subjects, 2, n_timesteps)``.
        y_train: Training labels.
    """
    channel = best_result["channel"]
    length = best_result["length"]
    shapelet_norm = best_shapelet["values_norm"]

    class0_subjects = np.where(y_train == 0)[0]
    class1_subjects = np.where(y_train == 1)[0]
    samples = [class0_subjects[0], class0_subjects[1],
               class1_subjects[0], class1_subjects[1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for axis, subject_index in zip(axes, samples):
        series = X_train[subject_index, channel]
        distance, match_start = subsequence_dist_znorm(series, shapelet_norm)

        # Raw values are drawn so the shape stays recognisable; only the match
        # location and the reported distance come from the normalised metric.
        axis.plot(series, color="gray", alpha=0.6,
                  label=f"Subject {subject_index} "
                        f"(Class {y_train[subject_index]})")
        axis.plot(range(match_start, match_start + length),
                  series[match_start:match_start + length],
                  color="red", linewidth=3, label="Best Shapelet Match")
        axis.set_title(f"Subject {subject_index} | "
                       f"Class {y_train[subject_index]} | "
                       f"Z-Norm Distance: {distance:.4f}")
        axis.legend()
        axis.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(f"Top Shapelet (ID {best_result['id']}, Channel {channel}, "
                 f"Length {length}) Local Matching Alignment", fontsize=16)
    plt.tight_layout()

    os.makedirs(PATHS["figures_dir"], exist_ok=True)
    plt.savefig(os.path.join(PATHS["figures_dir"],
                             "N4_Shapelet_Alignment_Overlay.png"))
    plt.show()
