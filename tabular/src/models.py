"""
STEP 4 — Model definitions.

Two things live here:

1. A small PyTorch multi-layer perceptron (MLP) wrapped so that scikit-learn
   can drive it like any other estimator — ``fit`` / ``predict`` /
   ``predict_proba``. PyTorch is used *only* for this; everything else in the
   project is NumPy, pandas and scikit-learn.
2. ``build_model()``, which turns a dictionary of tuned hyperparameters into a
   ready-to-fit estimator, optionally preceded by a PCA step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.config import SEEDS
from src.utils import cast_integer_params

DEFAULT_HIDDEN_LAYERS = [128, 64]

# Activation functions the tuner is allowed to pick from.
ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}

# Optimizers the tuner is allowed to pick from. Unknown names fall back to Adam.
OPTIMIZERS = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


# ---------------------------------------------------------------------------
# The network itself
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """A plain feed-forward network: [Linear -> activation -> dropout] x N -> Linear.

    Args:
        input_dim: Number of input features.
        output_dim: 1 for regression, or the number of classes.
        hidden_layers: Width of each hidden layer, e.g. ``[128, 64]``.
        activation: Key into ``ACTIVATIONS``.
        dropout: Dropout probability; no dropout layer is added when 0.
    """

    def __init__(self, input_dim, output_dim, hidden_layers=None,
                 activation="relu", dropout=0.0):
        super().__init__()
        # `is None` rather than a falsy check, so an explicitly empty list still
        # means "no hidden layers" rather than silently becoming the default.
        if hidden_layers is None:
            hidden_layers = DEFAULT_HIDDEN_LAYERS

        layers = []
        previous_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(previous_dim, width))
            if activation in ACTIVATIONS:
                layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = width
        layers.append(nn.Linear(previous_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Run a forward pass. Returns raw scores (logits for classification)."""
        return self.network(x)


# ---------------------------------------------------------------------------
# scikit-learn wrapper
# ---------------------------------------------------------------------------

class TorchEstimator(BaseEstimator):
    """Trains an :class:`MLP` behind a scikit-learn-style ``fit`` interface.

    Use the :class:`TorchClassifier` / :class:`TorchRegressor` subclasses.

    Args:
        hidden_layers: Hidden layer widths; defaults to ``[128, 64]``.
        activation: Key into ``ACTIVATIONS``.
        dropout: Dropout probability.
        lr: Learning rate.
        weight_decay: L2 penalty applied by the optimizer.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        device: ``"cpu"``, ``"mps"`` or ``"cuda"``.
        task: ``"regression"`` or ``"classification"``.
        verbose: Print the loss every 10 epochs.
        early_stopping_patience: Stop after this many epochs without a new best
            *training* loss.
        optimizer_name: Key into ``OPTIMIZERS``.
        **kwargs: Accepts ``optimizer=`` as an alias for ``optimizer_name``,
            which is how the tuned parameters arrive from the Optuna CSVs.
    """

    def __init__(self, hidden_layers=None, activation="relu", dropout=0.0,
                 lr=1e-3, weight_decay=1e-5, epochs=100, batch_size=32,
                 device="cpu", task="regression", verbose=False,
                 early_stopping_patience=10, optimizer_name="adamw", **kwargs):
        self.hidden_layers = (DEFAULT_HIDDEN_LAYERS if hidden_layers is None
                              else hidden_layers)
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.task = task
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        # Optuna records this hyperparameter under the name "optimizer", so the
        # tuned dictionaries arrive with that key rather than "optimizer_name".
        self.optimizer_name = kwargs.pop("optimizer", optimizer_name)
        self.model = None

    def _build_optimizer(self, parameters):
        """Return the configured torch optimizer over ``parameters``."""
        optimizer_class = OPTIMIZERS.get(self.optimizer_name.lower(), optim.Adam)
        return optimizer_class(parameters, lr=self.lr,
                               weight_decay=self.weight_decay)

    def fit(self, X, y, sample_weight=None):
        """Train the network.

        Args:
            X: 2D numeric array of features.
            y: Target vector.
            sample_weight: Accepted for scikit-learn compatibility (the CV
                helper passes it to every estimator) and ignored. For
                classification, imbalance is instead handled by the weighted
                cross-entropy loss built below; regression uses plain MSE.

        Returns:
            self.
        """
        X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)

        if self.task == "classification":
            # Map arbitrary class labels onto 0..n-1 for CrossEntropyLoss.
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self.device)
            output_dim = len(self.label_encoder.classes_)

            # Weight each class inversely to its frequency so the rare severe
            # cases are not drowned out by the majority "None" class.
            weights = class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_encoded), y=y_encoded
            )
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32).to(self.device)
            )
        else:
            y_tensor = torch.tensor(y.astype(np.float32)).to(self.device).view(-1, 1)
            output_dim = 1
            criterion = nn.MSELoss()

        loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=self.batch_size, shuffle=True)

        self.model = MLP(X.shape[1], output_dim, self.hidden_layers,
                         self.activation, self.dropout).to(self.device)
        self.optimizer = self._build_optimizer(self.model.parameters())

        self.model.train()
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            average_loss = epoch_loss / len(loader)

            # Early stopping watches the TRAINING loss (there is no validation
            # split here), so it caps wasted compute once the fit plateaus
            # rather than guarding against overfitting.
            if average_loss < best_loss:
                best_loss = average_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss:.4f}")

        self.is_fitted_ = True
        return self

    def _forward(self, X):
        """Run the trained network over ``X`` in eval mode, without gradients."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
            return self.model(X_tensor)


class TorchClassifier(TorchEstimator, ClassifierMixin):
    """MLP classifier.

    The ``**kwargs``-only signature is deliberate and load-bearing. Because
    ``BaseEstimator.get_params()`` reads the explicit arguments of ``__init__``,
    it returns an empty dict here — which means scikit-learn's ``clone()``
    rebuilds this estimator with *default* hyperparameters. That is exactly
    what happens to the MLP inside ``StackingClassifier``, and it is how the
    published stacking results were produced, so the signature is preserved
    as-is. See the "Known quirks" section of the README.
    """

    def __init__(self, **kwargs):
        super().__init__(task="classification", **kwargs)

    @property
    def classes_(self):
        """The original class labels, or None before ``fit``."""
        if hasattr(self, "label_encoder"):
            return self.label_encoder.classes_
        return None

    def predict(self, X):
        """Return predicted class labels in the original label space."""
        predicted = torch.max(self._forward(X), 1)[1]
        return self.label_encoder.inverse_transform(predicted.cpu().numpy())

    def predict_proba(self, X):
        """Return class probabilities, shape (n_samples, n_classes)."""
        return torch.softmax(self._forward(X), dim=1).cpu().numpy()


class TorchRegressor(TorchEstimator, RegressorMixin):
    """MLP regressor. Same ``**kwargs`` caveat as :class:`TorchClassifier`."""

    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)

    def predict(self, X):
        """Return continuous predictions as a flat 1D array."""
        return self._forward(X).cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Turning tuned hyperparameters into estimators
# ---------------------------------------------------------------------------

def build_model(model_class, tuned_params, pca_n=None, fixed_params=None):
    """Instantiate a model from tuned parameters, optionally behind a PCA step.

    Steps taken, in order:

    1. Drop keys that describe the pipeline rather than the model.
    2. Cast hyperparameters that must be ints (a CSV round-trip floats them).
    3. Apply the LogisticRegression solver rule, if relevant.
    4. Merge in ``fixed_params``, which always win.
    5. Wrap everything in a ``Pipeline``, with a leading PCA step when
       ``pca_n`` is set.

    The result is always a Pipeline, even with no PCA. A one-step Pipeline
    behaves identically to the bare estimator, and it means callers can always
    address the estimator as ``"model"`` — which the XGBoost stage relies on
    when it passes ``model__sample_weight`` through ``fit``.

    Args:
        model_class: Estimator class, e.g. ``lgb.LGBMClassifier``.
        tuned_params: Best hyperparameters from the Optuna study.
        pca_n: PCA component count, or None for no PCA step.
        fixed_params: Non-tuned keyword arguments (seeds, verbosity, weighting).

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.
    """
    params = dict(tuned_params)

    # These configure preprocessing or the tuner, not the estimator.
    for key in ["pca_n_components", "n_components", "imputer_choice"]:
        params.pop(key, None)

    params = cast_integer_params(params)

    # A CSV round-trip turns every numeric hyperparameter into a numpy scalar
    # (np.float64, np.int64). Most estimators tolerate that, but CatBoost's
    # constructor converts e.g. np.float64 -> float internally, so its
    # get_params() no longer matches the constructor arguments and sklearn's
    # clone() raises "constructor either does not set or modifies parameter".
    # Casting every value to a native Python type up front avoids that.
    params = {k: (v.item() if hasattr(v, "item") else v)
              for k, v in params.items()}

    if model_class is LogisticRegression:
        # A None penalty means "no regularisation", which makes C meaningless.
        if params.get("penalty") is None and "C" in params:
            params.pop("C")
        # Only liblinear supports an L1 penalty; the tuner searches L2 only,
        # so in practice this always resolves to lbfgs.
        params["solver"] = "liblinear" if params.get("penalty") == "l1" else "lbfgs"

    if fixed_params:
        # An explicit fixed value overrides whatever the tuner recorded.
        for key in fixed_params:
            params.pop(key, None)
        params.update(fixed_params)

    steps = []
    if pca_n:
        steps.append(("pca", PCA(n_components=int(pca_n),
                                 random_state=SEEDS["pca"])))
    steps.append(("model", model_class(**params)))
    return Pipeline(steps)
