"""Small torch MLP IDS classifier with an exposed embedding layer.

The model is trained ONLY on the known classes (benign, portscan, brute-force,
malware-c2). Its penultimate layer - the 32-d ReLU embedding - is the space in
which the Mahalanobis OOD scorer is fitted, and its logits feed the
energy-based OOD score.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import N_FEATURES, N_KNOWN


class IDSMLP(torch.nn.Module):
    """Two-hidden-layer MLP: features -> 64 -> 32 (embedding) -> n_known."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden: int = 64,
        embed: int = 32,
        n_classes: int = N_KNOWN,
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(n_features, hidden)
        self.fc2 = torch.nn.Linear(hidden, embed)
        self.head = torch.nn.Linear(embed, n_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer features used for Mahalanobis OOD scoring."""
        h = F.relu(self.fc1(x))
        return F.relu(self.fc2(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(x))


def _to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(X, dtype=np.float32))


def predict_logits(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(X)).numpy()


def predict_proba(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return F.softmax(model(_to_tensor(X)), dim=1).numpy()


def get_embeddings(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.embedding(_to_tensor(X)).numpy()


def train_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    verbose: bool = True,
) -> list[float]:
    """Train the IDS classifier on KNOWN classes (Adam, cross-entropy)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    Xt = _to_tensor(X)
    yt = torch.tensor(np.asarray(y, dtype=np.int64))
    n = Xt.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        ep_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)
        if verbose and (ep % 10 == 9 or ep == epochs - 1):
            print(f"  epoch {ep + 1}/{epochs}  loss={avg:.4f}")
    return history
