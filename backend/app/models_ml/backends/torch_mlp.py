"""PyTorch multilayer-perceptron land-cover backend.

An alternative to the tree ensemble on the same feature contract. It is useful
where a differentiable model is wanted (fine-tuning, embedding reuse, GPU
inference over large mosaics) and it trains on standardised inputs with class
weighting to counter label imbalance.

The artifact stores only tensors and plain configuration, so it loads with
``weights_only=True`` and does not execute code from the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from app.core.errors import ModelInferenceError
from app.core.logging import get_logger
from app.models_ml.backends.base import LandCoverBackend
from app.models_ml.labels import CLASS_ORDER

logger = get_logger(__name__)

DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "hidden_sizes": (128, 64),
    "dropout": 0.15,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 4096,
    "epochs": 30,
    "random_state": 42,
}


class _MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_sizes, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for size in hidden_sizes:
            layers += [
                nn.Linear(previous, size),
                nn.BatchNorm1d(size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            previous = size
        layers.append(nn.Linear(previous, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPBackend(LandCoverBackend):
    backend_name = "torch_mlp"
    supports_probability = True

    def __init__(self, n_features: int | None = None, **overrides: Any) -> None:
        self._params = {**DEFAULT_HYPERPARAMETERS, **overrides}
        self._params["hidden_sizes"] = tuple(self._params["hidden_sizes"])
        self.n_features = n_features
        self.n_classes = len(CLASS_ORDER)
        self.model: _MLP | None = None
        # Standardisation statistics learned from the training split.
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- training -------------------------------------------------------
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        torch.manual_seed(int(self._params["random_state"]))
        self.n_features = int(features.shape[1])
        self.feature_mean = features.mean(axis=0)
        self.feature_std = np.clip(features.std(axis=0), 1e-6, None)

        x = torch.from_numpy(self._standardise(features)).float()
        y = torch.from_numpy(labels.astype("int64"))

        self.model = _MLP(
            self.n_features,
            self.n_classes,
            self._params["hidden_sizes"],
            float(self._params["dropout"]),
        ).to(self.device)

        counts = np.bincount(labels.astype("int64"), minlength=self.n_classes).astype("float64")
        weights = np.where(counts > 0, counts.sum() / (self.n_classes * np.maximum(counts, 1)), 0.0)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )
        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self._params["learning_rate"]),
            weight_decay=float(self._params["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=int(self._params["epochs"])
        )

        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(self._params["batch_size"]), shuffle=True, drop_last=False
        )

        self.model.train()
        for epoch in range(int(self._params["epochs"])):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimiser.step()
                total_loss += float(loss.item()) * batch_x.shape[0]
            scheduler.step()
            if epoch % 5 == 0 or epoch == int(self._params["epochs"]) - 1:
                logger.info(
                    "mlp_epoch", epoch=epoch, mean_loss=round(total_loss / max(1, len(dataset)), 5)
                )

    # --- inference -------------------------------------------------------
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ModelInferenceError("The multilayer-perceptron backend has not been trained.")
        if features.size == 0:
            return np.zeros((0, self.n_classes), dtype="float32")

        self.model.eval()
        outputs: list[np.ndarray] = []
        batch = int(self._params["batch_size"])
        standardised = self._standardise(features)
        with torch.no_grad():
            for start in range(0, standardised.shape[0], batch):
                chunk = (
                    torch.from_numpy(standardised[start : start + batch]).float().to(self.device)
                )
                logits = self.model(chunk)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype("float32")

    def _standardise(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise ModelInferenceError("Feature standardisation statistics are missing.")
        return ((features - self.feature_mean) / self.feature_std).astype("float32")

    # --- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        if self.model is None:
            raise ModelInferenceError("Refusing to save an untrained model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "feature_mean": torch.from_numpy(np.asarray(self.feature_mean, dtype="float32")),
                "feature_std": torch.from_numpy(np.asarray(self.feature_std, dtype="float32")),
                "hidden_sizes": list(self._params["hidden_sizes"]),
                "dropout": float(self._params["dropout"]),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TorchMLPBackend:
        try:
            # weights_only=True restricts unpickling to tensors and primitives.
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelInferenceError(
                "The neural-network artifact could not be loaded.", details={"path": path.name}
            ) from exc

        n_features = int(payload["n_features"])
        instance = cls(
            n_features=n_features,
            hidden_sizes=tuple(payload["hidden_sizes"]),
            dropout=float(payload["dropout"]),
        )
        instance.n_classes = int(payload["n_classes"])
        instance.model = _MLP(
            n_features,
            instance.n_classes,
            tuple(payload["hidden_sizes"]),
            float(payload["dropout"]),
        )
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device)
        instance.model.eval()
        instance.feature_mean = payload["feature_mean"].numpy()
        instance.feature_std = payload["feature_std"].numpy()
        return instance

    def hyperparameters(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self._params.items()}
