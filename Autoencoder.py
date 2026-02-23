import torch
import torch.nn as nn


class SourceBasedAE(nn.Module):
    """Simple source-based autoencoder with FIXED (known) profiles only.

    - Encoder is trainable: Linear(m -> k, bias=False)
    - Decoder/profiles F are fixed: provided via F_fixed (shape m x k)
    - Nonnegativity on contributions via ReLU

    Forward returns (X_hat, G, F) to preserve your existing call sites.
    """

    def __init__(self, m, k, F_fixed=None, n_fixed=None):
        super().__init__()
        self.m = int(m)
        self.k = int(k)

        if F_fixed is None:
            raise ValueError("F_fixed (known profiles) must be provided")

        F = torch.as_tensor(F_fixed, dtype=torch.float64)
        if F.ndim != 2:
            raise ValueError(f"F_fixed must be 2D (m,k). Got {tuple(F.shape)}")

        if F.shape[0] != self.m:
            raise ValueError(f"F_fixed has m={F.shape[0]} but model m={self.m}")

        # If n_fixed provided in older code paths, enforce fixed-only behavior
        if n_fixed is not None and int(n_fixed) != F.shape[1]:
            raise ValueError(
                f"n_fixed must equal number of known profiles (k={F.shape[1]}). Got n_fixed={n_fixed}"
            )

        if self.k != F.shape[1]:
            raise ValueError(f"k must equal number of known profiles (k={F.shape[1]}). Got k={self.k}")

        self.register_buffer("F_fixed", F)

        # Trainable encoder
        self.encoder = nn.Linear(self.m, self.k, bias=False, dtype=torch.float64)

    def forward(self, X):
        z = self.encoder(X)
        G = torch.relu(z)
        F = self.F_fixed
        X_hat = G @ F.T
        return X_hat, G, F
