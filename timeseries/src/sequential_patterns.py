"""
STEP 6 — Sequential pattern mining.

The idea: turn each child's continuous activity trace into a short word made of
three letters — **L**ow, **M**edium, **H**igh activity — and then ask which
letter sequences show up often, and which show up in one class but not the
other.

The pipeline is:

1. **Discretise** (:func:`discretize_sequences`) — average the 200 time steps
   into 20 blocks, then label each block L/M/H by global tercile.
2. **Mine** (:func:`gsp`) — find every frequent contiguous pattern with the
   Generalized Sequential Patterns algorithm.
3. **Reduce** (:func:`filter_maximal_patterns`) — keep only maximal patterns,
   since a frequent pattern's every sub-pattern is frequent too and reporting
   them all is noise.
4. **Contrast** (:func:`run_contrast_mining`) — mine each class separately and
   keep the patterns unique to one of them.

``max_gap=1`` throughout means pattern elements must be strictly **contiguous**
in time — "low, then low, then low" describes three consecutive blocks, not
three low blocks scattered across the day.
"""

import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.config import (MINUTES_PER_STEP, SPM_MAX_GAP, SPM_MIN_SUPPORT,
                        SPM_PAA_WINDOW, SPM_QUANTILES, SPM_SYMBOLS, SPM_TOP_N)


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def discretize_sequences(X_channel, window_size=SPM_PAA_WINDOW):
    """Turn continuous signals into strings of L / M / H symbols.

    Two steps. First PAA: each block of ``window_size`` consecutive time steps
    collapses to its mean, cutting 200 steps to 20 and smoothing away noise.
    Then binning: the 33rd and 66th percentiles of *all* block averages across
    *all* subjects become the two cut points, so the three symbols are equally
    frequent overall. Using global rather than per-subject cut points is what
    makes an "H" mean the same thing for every child.

    Args:
        X_channel: Array of shape ``(n_subjects, n_timesteps)`` for one channel.
        window_size: Number of time steps averaged into one block.

    Returns:
        Tuple ``(sequences, quantiles)`` where ``sequences`` is a list of
        symbol lists and ``quantiles`` holds the two numeric cut points.
    """
    n_subjects = X_channel.shape[0]

    # PAA for every subject at once: reshape into blocks and average each.
    block_means = X_channel.reshape(n_subjects, -1, window_size).mean(axis=2)

    quantiles = np.percentile(block_means, SPM_QUANTILES)
    print(f"Computed Quantiles:\n  Low threshold (<33%): {quantiles[0]:.4f}\n"
          f"  High threshold (>66%): {quantiles[1]:.4f}")

    # searchsorted with side="right" reproduces the "v < low -> L,
    # v < high -> M, else H" rule exactly, including at the boundaries.
    symbol_indices = np.searchsorted(quantiles, block_means, side="right")
    symbols = np.array(SPM_SYMBOLS)
    sequences = [list(row) for row in symbols[symbol_indices]]

    print(f"Discretized {len(sequences)} sequences of length "
          f"{len(sequences[0])}.")
    print("\nSample sequence:")
    print(" -> ".join(sequences[0]))

    return sequences, quantiles


def print_symbol_distribution(sequences, labels):
    """Report how often each symbol occurs overall and within each class.

    A first, crude check for signal: if the classes had identical symbol mixes,
    no pattern built from those symbols could separate them.

    Args:
        sequences: List of symbol lists.
        labels: Binary label per sequence.
    """
    all_symbols = [symbol for sequence in sequences for symbol in sequence]
    counts = Counter(all_symbols)
    total = len(all_symbols)

    print("\n=== Global symbol distribution across all sequences ===")
    for symbol in SPM_SYMBOLS:
        count = counts.get(symbol, 0)
        print(f"  {symbol}: {count} ({count / total * 100:.1f}%)")

    class0_symbols = [symbol for sequence, label in zip(sequences, labels)
                      if label == 0 for symbol in sequence]
    class1_symbols = [symbol for sequence, label in zip(sequences, labels)
                      if label == 1 for symbol in sequence]
    counts0 = Counter(class0_symbols)
    counts1 = Counter(class1_symbols)
    total0 = len(class0_symbols)
    total1 = len(class1_symbols)

    print("\n=== Symbol distribution by SII class ===")
    print(f"  {'Sym':<4} | {'SII=0 (non-prob)':>18} | "
          f"{'SII=1 (problematic)':>20}")
    for symbol in SPM_SYMBOLS:
        n0 = counts0.get(symbol, 0)
        n1 = counts1.get(symbol, 0)
        print(f"  {symbol:<4} | {n0:>7} ({n0 / total0 * 100:>5.1f}%) | "
              f"{n1:>7} ({n1 / total1 * 100:>5.1f}%)")


# ---------------------------------------------------------------------------
# The GSP algorithm
# ---------------------------------------------------------------------------

def is_subsequence(pattern, sequence, max_gap=SPM_MAX_GAP):
    """Check whether a pattern occurs inside a sequence under the gap limit.

    ``max_gap=1`` allows no skipping at all, so this reduces to asking whether
    ``pattern`` appears as a run of consecutive symbols. The search is written
    generally so a looser gap can be explored by changing one argument.

    Args:
        pattern: Candidate pattern, as a list of symbols.
        sequence: Data sequence to search.
        max_gap: Maximum allowed gap between consecutive matched elements.

    Returns:
        True if the pattern occurs in the sequence.
    """
    if not pattern:
        return True

    def match_from(pattern_position, sequence_position):
        """Try to match the rest of the pattern starting at this position."""
        if pattern_position == len(pattern):
            return True
        if sequence_position >= len(sequence):
            return False

        for offset in range(max_gap):
            index = sequence_position + offset
            if (index < len(sequence)
                    and sequence[index] == pattern[pattern_position]
                    and match_from(pattern_position + 1, index + 1)):
                return True
        return False

    # Try every position where the pattern's first symbol occurs.
    for start in range(len(sequence)):
        if sequence[start] == pattern[0] and match_from(1, start + 1):
            return True
    return False


def get_frequent_1_sequences(dataset, min_support_count):
    """Find the single symbols that appear in enough sequences.

    Support is counted per *sequence*, not per occurrence — a subject with ten
    "H" blocks still contributes one to H's support.

    Args:
        dataset: List of sequences.
        min_support_count: Minimum number of sequences a symbol must appear in.

    Returns:
        List of frequent one-element patterns.
    """
    counts = defaultdict(int)
    for sequence in dataset:
        for symbol in set(sequence):
            counts[symbol] += 1
    return [[symbol] for symbol, count in counts.items()
            if count >= min_support_count]


def generate_candidates(frequent_patterns):
    """Build candidate patterns of length k from frequent ones of length k-1.

    The GSP join step: two patterns combine when one's tail matches the other's
    head, so "L -> M" and "M -> H" produce "L -> M -> H". Length-2 candidates
    are a special case — every ordered pair of frequent symbols is a candidate.

    Under ``max_gap=1`` every contiguous sub-pattern of a candidate built this
    way is already known frequent, so no extra Apriori pruning step is needed.

    Args:
        frequent_patterns: Frequent patterns of length k-1.

    Returns:
        List of candidate patterns of length k.
    """
    if not frequent_patterns:
        return []

    target_length = len(frequent_patterns[0]) + 1
    if target_length == 2:
        return [first + second
                for first in frequent_patterns
                for second in frequent_patterns]

    return [first + [second[-1]]
            for first in frequent_patterns
            for second in frequent_patterns
            if first[1:] == second[:-1]]


def gsp(dataset, min_support_ratio, max_gap=SPM_MAX_GAP):
    """Run Generalized Sequential Pattern mining, level by level.

    Starts from single frequent symbols and grows patterns one symbol at a time,
    stopping when no candidate of the next length is frequent enough.

    Args:
        dataset: List of sequences.
        min_support_ratio: Minimum support as a fraction of the dataset.
        max_gap: Maximum gap between consecutive matched elements.

    Returns:
        List of ``(pattern, count, support)`` tuples across all lengths.
    """
    min_support_count = int(len(dataset) * min_support_ratio)
    frequent_patterns = get_frequent_1_sequences(dataset, min_support_count)
    all_frequent = []

    while frequent_patterns:
        # Record this level with its exact support before growing further.
        for pattern in frequent_patterns:
            count = sum(1 for sequence in dataset
                        if is_subsequence(pattern, sequence, max_gap))
            all_frequent.append((pattern, count, count / len(dataset)))

        # Grow to the next length and keep only the candidates that qualify.
        frequent_patterns = [
            candidate for candidate in generate_candidates(frequent_patterns)
            if sum(1 for sequence in dataset
                   if is_subsequence(candidate, sequence, max_gap))
            >= min_support_count
        ]

    return all_frequent


def filter_maximal_patterns(patterns, max_gap=SPM_MAX_GAP):
    """Keep only patterns that are not contained in a longer frequent pattern.

    Every sub-pattern of a frequent pattern is itself frequent, so the raw
    output is dominated by fragments of a handful of long patterns. Keeping
    only the maximal ones removes that redundancy without losing information.

    Args:
        patterns: List of ``(pattern, count, support)`` tuples.
        max_gap: Gap constraint used for the containment test.

    Returns:
        List of maximal ``(pattern, count, support)`` tuples.
    """
    # Longest first, so any pattern is only ever tested against patterns that
    # could actually contain it.
    patterns = sorted(patterns, key=lambda entry: len(entry[0]), reverse=True)

    maximal = []
    for entry in patterns:
        if not any(is_subsequence(entry[0], kept[0], max_gap)
                   for kept in maximal):
            maximal.append(entry)
    return maximal


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_global_patterns(sequences, min_support_ratio=SPM_MIN_SUPPORT,
                           max_gap=SPM_MAX_GAP):
    """Mine the whole dataset and print the maximal patterns it contains.

    Args:
        sequences: List of symbol sequences.
        min_support_ratio: Minimum support fraction.
        max_gap: Gap constraint.

    Returns:
        List of maximal ``(pattern, count, support)`` tuples, best support
        first.
    """
    print(f"Running contiguous GSP (max_gap={max_gap}) with minimum support: "
          f"{min_support_ratio * 100:.0f}%")
    start_time = time.time()

    frequent = gsp(sequences, min_support_ratio, max_gap)
    print(f"Extracted {len(frequent)} raw frequent patterns.")

    maximal = filter_maximal_patterns(frequent, max_gap)
    print(f"Reduced to {len(maximal)} Maximal Sequential Patterns in "
          f"{time.time() - start_time:.2f} seconds.")

    maximal.sort(key=lambda entry: entry[1], reverse=True)

    print(f"\nTop {SPM_TOP_N} Most Supported Contiguous Maximal Patterns:")
    for pattern, count, support in maximal[:SPM_TOP_N]:
        print(f"  {' -> '.join(pattern):<25} | Supp: {support:.1%} "
              f"(n={count})")

    print("\n=== Maximal pattern length distribution ===")
    lengths = [len(entry[0]) for entry in maximal]
    length_counts = Counter(lengths)
    for length in sorted(length_counts):
        print(f"  Length {length}: {length_counts[length]} patterns")
    print(f"  Longest pattern: length {max(lengths)}")
    print(f"  Shortest pattern: length {min(lengths)}")

    print("\n=== All maximal patterns (sorted by support) ===")
    for pattern, count, support in maximal:
        print(f"  {' -> '.join(pattern):<35} | {support:.1%} (n={count})")

    return maximal


def run_contrast_mining(sequences, labels, min_support_ratio=SPM_MIN_SUPPORT,
                        max_gap=SPM_MAX_GAP):
    """Mine each class separately and isolate the patterns unique to each.

    A pattern that is frequent in both cohorts describes ordinary behaviour and
    tells us nothing. The interesting patterns are the ones that clear the
    support bar in one cohort and not the other.

    Args:
        sequences: List of symbol sequences.
        labels: Binary label per sequence.
        min_support_ratio: Minimum support fraction, applied within each cohort.
        max_gap: Gap constraint.

    Returns:
        Tuple ``(unique_to_0, unique_to_1, support_0, support_1,
        maximal_0, maximal_1)``. The ``unique_*`` entries are sets of pattern
        tuples; the ``support_*`` entries map pattern tuples to their support.
    """
    class0_sequences = [sequence for sequence, label in zip(sequences, labels)
                        if label == 0]
    class1_sequences = [sequence for sequence, label in zip(sequences, labels)
                        if label == 1]

    print(f"Cohort sizes:\n  SII=0 (Non-problematic): "
          f"{len(class0_sequences)}\n  SII=1 (Problematic):     "
          f"{len(class1_sequences)}")
    print(f"\nRunning Contiguous GSP (min_sup="
          f"{min_support_ratio * 100:.0f}%, max_gap={max_gap})...")

    maximal_0 = filter_maximal_patterns(
        gsp(class0_sequences, min_support_ratio, max_gap), max_gap)
    maximal_1 = filter_maximal_patterns(
        gsp(class1_sequences, min_support_ratio, max_gap), max_gap)
    print(f"Maximal patterns found:\n  SII=0: {len(maximal_0)}\n"
          f"  SII=1: {len(maximal_1)}")

    set_0 = {tuple(entry[0]) for entry in maximal_0}
    set_1 = {tuple(entry[0]) for entry in maximal_1}
    unique_to_0 = set_0 - set_1
    unique_to_1 = set_1 - set_0

    support_0 = {tuple(entry[0]): entry[2] for entry in maximal_0}
    support_1 = {tuple(entry[0]): entry[2] for entry in maximal_1}

    print(f"\nDiscriminative Maximal Patterns:\n  Unique to SII=0: "
          f"{len(unique_to_0)}\n  Unique to SII=1: {len(unique_to_1)}")

    return (unique_to_0, unique_to_1, support_0, support_1, maximal_0,
            maximal_1)


def rank_unique_patterns(unique_patterns, supports, top_n=SPM_TOP_N):
    """Rank a cohort's unique patterns by length first, then by support.

    Length leads because a long contiguous pattern describes a longer stretch
    of sustained behaviour, which is the clinically interesting signal here.

    Args:
        unique_patterns: Set of pattern tuples unique to one cohort.
        supports: Mapping from pattern tuple to support.
        top_n: How many to return.

    Returns:
        List of the top pattern tuples.
    """
    return sorted(unique_patterns,
                  key=lambda pattern: (len(pattern), supports[pattern]),
                  reverse=True)[:top_n]


def print_contrast_report(unique_to_0, unique_to_1, support_0, support_1):
    """Print the discriminative patterns for each cohort and the shared ones.

    Args:
        unique_to_0: Patterns unique to the non-problematic cohort.
        unique_to_1: Patterns unique to the problematic cohort.
        support_0: Support lookup for cohort 0.
        support_1: Support lookup for cohort 1.

    Returns:
        Tuple ``(top_class0, top_class1)`` of the ranked pattern lists.
    """
    top_class0 = rank_unique_patterns(unique_to_0, support_0)
    top_class1 = rank_unique_patterns(unique_to_1, support_1)

    print("\n=== Top discriminative patterns unique to SII=0 "
          "(non-problematic) ===")
    for pattern in top_class0:
        print(f"  {' -> '.join(pattern):<35} | support in SII=0: "
              f"{support_0[pattern]:.1%}")

    print("\n=== Top discriminative patterns unique to SII=1 "
          "(problematic) ===")
    for pattern in top_class1:
        print(f"  {' -> '.join(pattern):<35} | support in SII=1: "
              f"{support_1[pattern]:.1%}")

    print("\n=== Patterns appearing in BOTH groups (non-discriminative) ===")
    shared = set(support_0) & set(support_1)
    most_shared = sorted(
        shared,
        key=lambda pattern: support_0.get(pattern, 0) + support_1.get(pattern, 0),
        reverse=True)[:5]
    for pattern in most_shared:
        print(f"  {' -> '.join(pattern):<35} | "
              f"SII=0: {support_0.get(pattern, 0):.1%}  "
              f"SII=1: {support_1.get(pattern, 0):.1%}")

    return top_class0, top_class1


def plot_contrast_patterns(top_class0, top_class1, support_0, support_1,
                           output_path, min_support_ratio=SPM_MIN_SUPPORT):
    """Draw the top discriminative patterns for both cohorts as bar charts.

    The x-axis starts at the minimum support rather than at zero, so the
    differences between patterns that all cleared the same bar stay visible.

    Args:
        top_class0: Ranked patterns unique to cohort 0.
        top_class1: Ranked patterns unique to cohort 1.
        support_0: Support lookup for cohort 0.
        support_1: Support lookup for cohort 1.
        output_path: Where to save the figure.
        min_support_ratio: Used to set the left edge of the x-axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    panels = [
        (axes[0], top_class0, support_0, "royalblue",
         "Contiguous Maximal Patterns: SII=0 (Non-problematic)"),
        (axes[1], top_class1, support_1, "crimson",
         "Contiguous Maximal Patterns: SII=1 (Problematic)"),
    ]

    for axis, patterns, supports, color, title in panels:
        percentages = [supports[pattern] * 100 for pattern in patterns]
        positions = np.arange(len(patterns))

        axis.barh(positions, percentages, color=color, edgecolor="black")
        axis.set_yticks(positions)
        axis.set_yticklabels([" -> ".join(pattern) for pattern in patterns],
                             fontsize=10)
        axis.invert_yaxis()   # highest-ranked pattern at the top
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("Support (%)")
        if percentages:
            axis.set_xlim(min_support_ratio * 100, max(percentages) + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def print_pattern_interpretation(top_class0, top_class1, support_0, support_1,
                                 quantiles, window_size=SPM_PAA_WINDOW):
    """Translate the longest unique pattern of each cohort into real durations.

    Each symbol covers ``window_size`` time steps of ``MINUTES_PER_STEP``
    minutes, so a six-symbol pattern describes a genuinely long stretch of
    uninterrupted behaviour rather than a momentary dip.

    Args:
        top_class0: Ranked patterns unique to cohort 0.
        top_class1: Ranked patterns unique to cohort 1.
        support_0: Support lookup for cohort 0.
        support_1: Support lookup for cohort 1.
        quantiles: The two numeric L/M/H cut points.
        window_size: Time steps averaged into one symbol.
    """
    minutes_per_symbol = window_size * MINUTES_PER_STEP

    print("\n=== Computed interpretation of discriminative patterns ===")
    if top_class1:
        longest = top_class1[0]
        hours = len(longest) * minutes_per_symbol / 60
        print(f"  Longest SII=1-unique pattern: '{' -> '.join(longest)}'")
        print(f"  Length: {len(longest)} segments × {minutes_per_symbol:.0f} "
              f"min/segment = {hours:.1f}h of continuous behaviour")
        print(f"  Support: {support_1[longest]:.1%} of problematic children "
              "show this pattern")

    if top_class0:
        longest = top_class0[0]
        hours = len(longest) * minutes_per_symbol / 60
        print(f"\n  Longest SII=0-unique pattern: '{' -> '.join(longest)}'")
        print(f"  Length: {len(longest)} segments × {minutes_per_symbol:.0f} "
              f"min/segment = {hours:.1f}h of continuous behaviour")
        print(f"  Support: {support_0[longest]:.1%} of non-problematic "
              "children show this pattern")

    print(f"\n  Quantile thresholds: L < {quantiles[0]:.4f} g  ≤  M < "
          f"{quantiles[1]:.4f} g  ≤  H (ENMO in raw units)")
