import argparse
import torch
import numpy as np

from Measurements import Measurement
from Autoencoder import SourceBasedAE



def init_encoder_from_F(F):
    """Return W0 (k x m) such that X_hat = relu(X @ W0^T) @ F^T is a sensible init."""
    FtF = F.T @ F
    W0 = np.linalg.pinv(FtF) @ F.T
    return W0


def compute_loss(X, X_hat, E, eps=1e-6):
    if isinstance(E, np.ndarray):
        E = torch.tensor(E, dtype=torch.float64, device=X.device)
    E_safe = torch.clamp(E, min=eps)
    R = (X - X_hat) / E_safe
    return torch.mean(R * R)


def train_fixed_profile_source_ae(meas: Measurement, lr=1e-2, epochs=500):
    """Train ONLY the encoder of a SourceBasedAE with FIXED known profiles."""
    X = torch.tensor(meas.get_X(), dtype=torch.float64)
    E = torch.tensor(meas.get_error(), dtype=torch.float64)

    F_fixed = meas.get_F_fixed()
    if F_fixed is None:
        raise ValueError("Measurement.get_F_fixed() returned None. Load fixed profiles first.")

    m = X.shape[1]
    k_fixed = F_fixed.shape[1]

    model = SourceBasedAE(m, k_fixed, F_fixed=F_fixed, n_fixed=k_fixed).double()

    # init encoder from known profiles
    with torch.no_grad():
        W0 = init_encoder_from_F(F_fixed)
        model.encoder.weight.copy_(torch.tensor(W0, dtype=torch.float64))

    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        X_hat, G, F = model(X)
        loss = compute_loss(X, X_hat, E)

        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss at epoch {epoch}")

        loss.backward()
        optimizer.step()

    return model


def main():
    torch.set_default_dtype(torch.float64)

    parser = argparse.ArgumentParser(description="Train a fixed-profile SourceBasedAE (single run).")

    parser.add_argument("--input", required=True, help="Path to input .xlsx or .csv file.")
    parser.add_argument("--output", required=True, help="Output prefix (used by Measurement).")

    parser.add_argument("--fixed_profiles", required=True,
                        help="Path to .xlsx containing known profiles (library).")

    parser.add_argument(
        "--fixed_labels", nargs="+", default=["HOA", "CCOA", "BBOA"],
        help="Fixed profile labels to use (space-separated)."
    )

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--random_fixed_profiles", action="store_true")

    args = parser.parse_args()

    meas = Measurement(input_path=args.input,
                       F_fixed_path=args.fixed_profiles,
                       output_prefix=args.output,
                       plot_subdir="SourceAE")

    meas.load()
    meas.load_fixed_profiles(labels=tuple(args.fixed_labels),
                             random_fixed=args.random_fixed_profiles)

    print("\nTraining fixed-profile SourceBasedAE (single run).")

    model = train_fixed_profile_source_ae(
        meas, lr=args.lr, epochs=args.epochs
    )

    with torch.no_grad():
        X = torch.tensor(meas.get_X(), dtype=torch.float64)
        X_hat, G, F = model(X)

    F_np = F.detach().cpu().numpy()
    G_np = G.detach().cpu().numpy()
    X_hat_np = X_hat.detach().cpu().numpy()

    # -------------------------
    # Measurement outputs (unchanged behavior)
    # -------------------------
    meas.set_F(F_np)
    meas.set_G(G_np)

    meas.Excel_results_creation()
    meas.plot_all()
    meas.compare_to_ground_truth()


    # RSS diagnostics
    rss_model = meas.compute_rss(X_hat_np)

    if meas.has_ground_truth():
        X_gt = meas.get_G_truth() @ meas.get_F_truth().T
        rss_gt = meas.compute_rss(X_gt)

        print("\nRSS diagnostics (Fixed-profile Source-Based AE):")
        print(f"  RSS model      = {rss_model:.3e}")
        print(f"  RSS groundtruth= {rss_gt:.3e}")
        print(f"  RSS ratio      = {rss_model / rss_gt:.3f}")

        X_np = meas.get_X()
        E_np = meas.get_error()

        rss_model_weighted = np.mean(((X_np - X_hat_np) / E_np) ** 2)
        rss_gt_weighted = np.mean(((X_np - X_gt) / E_np) ** 2)

        print("Weighted RSS model:", rss_model_weighted)
        print("Weighted RSS GT:", rss_gt_weighted)
        print("Ratio:", rss_model_weighted / rss_gt_weighted)
    else:
        print(f"\nRSS model (Fixed-profile Source-Based AE) = {rss_model:.3e}")

    print("Done: trained fixed-profile SourceBasedAE and saved plots/excels.")


if __name__ == "__main__":
    main()
