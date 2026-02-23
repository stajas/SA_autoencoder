import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import norm, gaussian_kde


class Measurement:
    """
    Handles:
      - Loading measurement data (X, time, error)
      - Loading ground-truth F/G when available
      - Loading fixed profiles from a library
      - Saving CSV outputs
      - Plotting (F profiles, G time series, stacked, correlations, diurnal)
      - Residual diagnostics
    """

    def __init__(self, input_path, F_fixed_path=None, X=None, time=None, mz_labels=None,
                 output_prefix="output", plot_subdir="plots"):

        self.input_path = input_path
        self.output_prefix = output_prefix

        self.plot_dir = os.path.join(self.output_prefix, plot_subdir)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Output CSVs
        self.output_F = os.path.join(self.plot_dir, f"{plot_subdir}_F_learned.csv")
        self.output_G = os.path.join(self.plot_dir, f"{plot_subdir}_G_learned.csv")
        self.output_RMSE = os.path.join(self.plot_dir, f"{plot_subdir}_RMSE.csv")

        # Plot files
        self.plot_F = os.path.join(self.plot_dir, f"F_profiles_{plot_subdir}.png")
        self.plot_G = os.path.join(self.plot_dir, f"G_profiles_{plot_subdir}.png")
        self.plot_G_diurnal_path = os.path.join(self.plot_dir, f"G_diurnal_{plot_subdir}.png")
        self.plot_stacked = os.path.join(self.plot_dir, f"stacked_contributions_{plot_subdir}.png")
        self.plot_corr = os.path.join(self.plot_dir, f"correlation_G_{plot_subdir}.png")

        # Data placeholders
        self.X = X
        self.time = time
        self.mz_labels = mz_labels
        self.E = None

        self.F_fixed_path = F_fixed_path
        self.F_fixed = None
        self.n_fixed = None

        self.F_truth = None
        self.G_truth = None
        self.F_learned = None
        self.G_learned = None

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @staticmethod
    def _parse_mz_labels(columns):
        mz_labels = []
        for col in columns:
            match = re.search(r"[-+]?\d*\.\d+|\d+", str(col))
            if match:
                mz_labels.append(float(match.group()))
            else:
                raise ValueError(f"Could not parse numeric m/z from column '{col}'")
        return np.array(mz_labels)

    @staticmethod
    def diurnal_mean(time, G):
        """
        Average across days at each hour-of-day.
        time: array-like datetime (n,)
        G: ndarray (n,k)
        returns hours (24,), mean (24,k), sem (24,k)
        """
        t = pd.to_datetime(time)
        df = pd.DataFrame(G, index=t)
        grp = df.groupby(df.index.hour)
        mean = grp.mean().reindex(range(24)).to_numpy()
        sem = (grp.std() / np.sqrt(grp.count())).reindex(range(24)).to_numpy()
        return np.arange(24), mean, sem

    @staticmethod
    def relative_profile_drift(F_runs, F_init, eps=1e-12):
        """
        F_runs: (n_runs, m, k)
        F_init: (m, k)
        returns drift: (n_runs, k)
        """
        F_runs = np.asarray(F_runs)
        F_init = np.asarray(F_init)
        n_runs, m, k = F_runs.shape
        drift = np.zeros((n_runs, k))
        denom = np.linalg.norm(F_init, axis=0) + eps  # (k,)

        for r in range(n_runs):
            num = np.linalg.norm(F_runs[r] - F_init, axis=0)
            drift[r] = num / denom
        return drift

    # ---------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------
    def load(self):
        xls = pd.ExcelFile(self.input_path)
        sheets = xls.sheet_names
        print(f"Available sheets: {sheets}")

        # ---- X sheet ----
        if "measurements" in sheets:
            df_X = pd.read_excel(self.input_path, sheet_name="measurements")
        elif "X" in sheets:
            df_X = pd.read_excel(self.input_path, sheet_name="X")
        else:
            raise ValueError("Could not find 'measurements' or 'X' sheet.")

        self.time = pd.to_datetime(df_X.iloc[:, 0], dayfirst=True)
        self.X = df_X.iloc[:, 1:].replace(",", ".", regex=True).astype(float).values
        self.mz_labels = self._parse_mz_labels(df_X.columns[1:])

        # ---- ground truth ----
        if "F" in sheets:
            df_F = pd.read_excel(self.input_path, sheet_name="F")
            self.F_truth = df_F.iloc[:, 1:].fillna(0).to_numpy()

        if "G" in sheets:
            df_G = pd.read_excel(self.input_path, sheet_name="G")
            self.G_truth = df_G.iloc[:, 1:].fillna(0).to_numpy()

        # ---- error ----
        if "error" in sheets:
            df_err = pd.read_excel(self.input_path, sheet_name="error")
            self.E = df_err.iloc[:, 1:].replace(",", ".", regex=True).astype(float).values
        else:
            self.E = None

        print("Data loaded successfully.")

    def has_ground_truth(self):
        return (self.F_truth is not None) and (self.G_truth is not None)

    def has_error(self):
        return self.E is not None

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------
    def compare_to_ground_truth(self):
        if not self.has_ground_truth():
            print("No ground truth available. Skipping comparison.")
            return

        F_truth = self.F_truth
        G_truth = self.G_truth
        k = self.G_learned.shape[1]

        print("\nComparison with Ground Truth (max correlation matching):")
        for i in range(k):
            corr_Gs = [np.corrcoef(self.G_learned[:, i], G_truth[:, j])[0, 1] for j in range(k)]
            corr_Fs = [np.corrcoef(self.F_learned[:, i], F_truth[:, j])[0, 1] for j in range(k)]

            jG = int(np.argmax(np.abs(corr_Gs)))
            jF = int(np.argmax(np.abs(corr_Fs)))

            rmse_G = np.sqrt(np.mean((self.G_learned[:, i] - G_truth[:, jG]) ** 2))
            rmse_F = np.sqrt(np.mean((self.F_learned[:, i] - F_truth[:, jF]) ** 2))

            print(
                f"Learned {i+1}: "
                f"G→Truth {jG+1} (corr={corr_Gs[jG]:.3f}, rmse={rmse_G:.3f}) | "
                f"F→Truth {jF+1} (corr={corr_Fs[jF]:.3f}, rmse={rmse_F:.3f})"
            )

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    def plot_all(self):
        """Convenience: save all standard plots."""
        self.plot_F_profiles()
        self.plot_G_timeseries()
        self.plot_G_stacked()
        self.plot_G_correlation()
        self.plot_G_diurnal()
        print(f"\nPlots saved to directory: {self.plot_dir}")

    def plot_F_profiles(self):
        if self.F_learned is None:
            raise ValueError("F_learned is None. Call set_F first.")
        k = self.F_learned.shape[1]

        fig, axs = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)
        width = 0.35

        for i in range(k):
            axs[i].bar(self.mz_labels, self.F_learned[:, i], width=width, label=f"Source {i+1}")
            if self.F_truth is not None:
                axs[i].bar(self.mz_labels + width, self.F_truth[:, i], width=width, alpha=0.7, label="Truth")
            axs[i].legend()
            axs[i].set_ylabel("Intensity")

        axs[-1].set_xlabel("m/z")
        formatted = [str(int(mz)) if float(mz).is_integer() else f"{mz:.2f}" for mz in self.mz_labels]
        axs[-1].set_xticks(self.mz_labels)
        axs[-1].set_xticklabels(formatted, fontsize=4, rotation=90)

        plt.tight_layout()
        plt.savefig(self.plot_F, dpi=300)
        plt.close()

    def plot_G_timeseries(self):
        if self.G_learned is None:
            raise ValueError("G_learned is None. Call set_G first.")
        k = self.G_learned.shape[1]

        fig, axs = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)
        for i in range(k):
            axs[i].plot(self.time, self.G_learned[:, i], label=f"Source {i+1}")
            if self.G_truth is not None:
                axs[i].plot(self.time, self.G_truth[:, i], "--", label="Truth")
            axs[i].legend()
            axs[i].set_ylabel("Contribution")

        axs[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.savefig(self.plot_G, dpi=300)
        plt.close()

    def plot_G_diurnal(self):
        """Diurnal average (mean across days per hour)."""
        if self.G_learned is None:
            raise ValueError("G_learned is None. Call set_G first.")
        k = self.G_learned.shape[1]

        hours, Gm, Gsem = self.diurnal_mean(self.time, self.G_learned)

        fig, axs = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)
        for i in range(k):
            axs[i].plot(hours, Gm[:, i], label=f"Source {i+1}")
            axs[i].fill_between(hours, Gm[:, i] - Gsem[:, i], Gm[:, i] + Gsem[:, i], alpha=0.2)

            if self.G_truth is not None:
                _, Gt_m, Gt_sem = self.diurnal_mean(self.time, self.G_truth)
                axs[i].plot(hours, Gt_m[:, i], "--", label="Truth")
                axs[i].fill_between(hours, Gt_m[:, i] - Gt_sem[:, i], Gt_m[:, i] + Gt_sem[:, i], alpha=0.2)

            axs[i].legend()
            axs[i].set_ylabel("Avg contrib")

        axs[-1].set_xlabel("Hour of day")
        axs[-1].set_xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(self.plot_G_diurnal_path, dpi=300)
        plt.close()

    def plot_G_stacked(self):
        if self.G_learned is None:
            raise ValueError("G_learned is None. Call set_G first.")
        k = self.G_learned.shape[1]

        plt.figure(figsize=(10, 4))
        plt.stackplot(self.time, self.G_learned.T, labels=[f"S{i+1}" for i in range(k)])
        plt.xlabel("Time")
        plt.ylabel("Contribution")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(self.plot_stacked, dpi=300)
        plt.close()

    def plot_G_correlation(self):
        if self.G_truth is None or self.G_learned is None:
            return
        k = self.G_learned.shape[1]

        fig, axs = plt.subplots(1, k, figsize=(4 * k, 4))
        if k == 1:
            axs = [axs]

        for i in range(k):
            axs[i].scatter(self.G_truth[:, i], self.G_learned[:, i], alpha=0.5)
            corr = np.corrcoef(self.G_truth[:, i], self.G_learned[:, i])[0, 1]
            axs[i].set_title(f"Source {i+1}\nr={corr:.2f}")
            axs[i].set_xlabel("Truth")
            axs[i].set_ylabel("Learned")

        plt.tight_layout()
        plt.savefig(self.plot_corr, dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Residual diagnostics
    # ---------------------------------------------------------
    def plot_scaled_residuals(self, X_hat):
        self.X = np.asarray(self.X)
        X_hat = np.asarray(X_hat)
        eps = 1e-9

        # relative (unsigned) residuals
        R_rel = np.abs((self.X - X_hat) / np.maximum(self.X, eps))

        path_res_mz = os.path.join(self.plot_dir, "residuals_over_mz.png")
        path_res_time = os.path.join(self.plot_dir, "residuals_over_time.png")
        path_res_hist = os.path.join(self.plot_dir, "residuals_histogram.png")

        # (a) over m/z
        mean_mz = R_rel.mean(axis=0)
        std_mz = R_rel.std(axis=0)

        plt.figure(figsize=(10, 4))
        plt.bar(self.mz_labels, mean_mz, yerr=std_mz, alpha=0.8, capsize=3)
        plt.xlabel("m/z")
        plt.ylabel("Scaled residual")
        plt.title("Scaled Residuals Over m/z (|X-Xhat|/X)")
        formatted = [str(int(float(mz))) if float(mz).is_integer() else str(mz) for mz in self.mz_labels]
        plt.xticks(self.mz_labels, formatted, fontsize=4)
        plt.tight_layout()
        plt.savefig(path_res_mz, dpi=300)
        plt.close()

        # (b) over time
        mean_t = R_rel.mean(axis=1)
        std_t = R_rel.std(axis=1)

        plt.figure(figsize=(10, 4))
        plt.plot(self.time, mean_t, label="Mean residual")
        plt.fill_between(self.time, mean_t - std_t, mean_t + std_t, alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel("Scaled residual")
        plt.title("Scaled Residuals Over Time")
        plt.tight_layout()
        plt.savefig(path_res_time, dpi=300)
        plt.close()

        # (c) signed scaled residuals histogram (requires E)
        if self.E is not None:
            R = (self.X - X_hat) / (self.E + eps)
            flat = R.ravel()

            p_low, p_high = np.percentile(flat, [1, 99])
            flat_trim = flat[(flat >= p_low) & (flat <= p_high)]

            plt.figure(figsize=(7, 5))
            plt.hist(flat_trim, bins=60, density=True, alpha=0.55, edgecolor="black", linewidth=0.4,
                     label="Residuals (1–99th pct)")

            mu, sigma = flat_trim.mean(), flat_trim.std()
            x = np.linspace(flat_trim.min(), flat_trim.max(), 300)
            plt.plot(x, norm.pdf(x, mu, sigma), linewidth=2, alpha=0.6, label="Normal fit")

            try:
                kde = gaussian_kde(flat_trim)
                plt.plot(x, kde(x), linewidth=2, alpha=0.6, label="KDE")
            except Exception:
                pass

            plt.xlabel("Scaled residual (signed) (X-Xhat)/E")
            plt.ylabel("Density")
            plt.title(f"Residual Distribution (μ={mu:.3f}, σ={sigma:.3f})")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path_res_hist, dpi=300)
            plt.close()

        print("\nResidual diagnostic plots saved:")
        print(f"  - {path_res_mz}")
        print(f"  - {path_res_time}")
        if self.E is not None:
            print(f"  - {path_res_hist}")

    # ---------------------------------------------------------
    # Fixed profiles
    # ---------------------------------------------------------
    def load_fixed_profiles(self, labels=("HOA", "CCOA", "BBOA"), random_fixed=False, rng_seed=42):
        """
        Load fixed source profiles from a library (multi-index columns: (label, variant)).
        Select one variant per label.
        """
        if self.F_fixed_path is None:
            raise ValueError("F_fixed_path is None.")

        print("Loading fixed source profiles (random selection per profile)...")

        df = pd.read_excel(self.F_fixed_path, sheet_name="Sheet1", decimal=",", header=[0, 1])
        df.columns = pd.MultiIndex.from_tuples([(str(c[0]).strip(), str(c[1]).strip()) for c in df.columns])

        mz_col = df.columns[0]
        df_filtered = df[df[mz_col].isin(self.mz_labels)]

        rng = np.random.default_rng(None if random_fixed else rng_seed)

        fixed_profiles = []
        for label in labels:
            matching_cols = [c for c in df.columns if c[0].lower() == label.lower()]
            if not matching_cols:
                raise ValueError(f"No columns found for label '{label}'")

            selected_col = matching_cols[rng.integers(len(matching_cols))]
            profile = pd.to_numeric(df_filtered[selected_col], errors="coerce").to_numpy()
            profile = np.nan_to_num(profile, nan=0.0)

            fixed_profiles.append(profile)
            print(f"Selected variant for {label}: column '{selected_col}'")

        F_fixed = np.stack(fixed_profiles, axis=1)  # (m, k_fixed)
        self.F_fixed = np.nan_to_num(F_fixed, nan=0.0)
        self.n_fixed = self.F_fixed.shape[1]

        print(f"Fixed profile matrix shape: {self.F_fixed.shape} (m × k_fixed)")
        
    def plot_G_with_uncertainty(self, time, G_center, G_lo, G_hi, out_path, labels=None):
    

        G_center = np.asarray(G_center)
        G_lo = np.asarray(G_lo)
        G_hi = np.asarray(G_hi)

        # ensure (n,k)
        if G_center.ndim == 1:
            G_center = G_center[:, None]
        if G_lo.ndim == 1:
            G_lo = G_lo[:, None]
        if G_hi.ndim == 1:
            G_hi = G_hi[:, None]

        k = G_center.shape[1]
        fig, axs = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)
        if k == 1:
            axs = [axs]

        if labels is None:
            labels = [f"Source {i+1}" for i in range(k)]

        for i in range(k):
            axs[i].plot(time, G_center[:, i], label=labels[i])
            axs[i].fill_between(time, G_lo[:, i], G_hi[:, i], alpha=0.2)
            axs[i].legend()
            axs[i].set_ylabel("Contribution")

        axs[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


    def plot_F_with_uncertainty(self, mz, F_center, F_lo, F_hi, out_path, labels=None):
        import numpy as np
        import matplotlib.pyplot as plt

        F_center = np.asarray(F_center)
        F_lo = np.asarray(F_lo)
        F_hi = np.asarray(F_hi)

        # ensure (m,k)
        if F_center.ndim == 1:
            F_center = F_center[:, None]
        if F_lo.ndim == 1:
            F_lo = F_lo[:, None]
        if F_hi.ndim == 1:
            F_hi = F_hi[:, None]

        k = F_center.shape[1]
        fig, axs = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)
        if k == 1:
            axs = [axs]

        if labels is None:
            labels = [f"Source {i+1}" for i in range(k)]

        for i in range(k):
            axs[i].plot(mz, F_center[:, i], label=labels[i])
            axs[i].fill_between(mz, F_lo[:, i], F_hi[:, i], alpha=0.2)
            axs[i].legend()
            axs[i].set_ylabel("Intensity")

        axs[-1].set_xlabel("m/z")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_scatter_G(self):
        if self.G_truth is None or self.G_learned is None:
            print("No G truth/learned. Skipping G scatter.")
            return

        k = self.G_learned.shape[1]
        out = os.path.join(self.plot_dir, "scatter_G_truth_vs_learned.png")

        fig, axs = plt.subplots(1, k, figsize=(4*k, 4))
        if k == 1:
            axs = [axs]

        for i in range(k):
            x = self.G_truth[:, i]
            y = self.G_learned[:, i]
            axs[i].scatter(x, y, alpha=0.35)
            r = np.corrcoef(x, y)[0, 1]
            axs[i].set_title(f"G Source {i+1}\nr={r:.2f}")
            axs[i].set_xlabel("Truth")
            axs[i].set_ylabel("Learned")

            # y=x reference line
            lo = np.nanmin([x.min(), y.min()])
            hi = np.nanmax([x.max(), y.max()])
            axs[i].plot([lo, hi], [lo, hi], "--", linewidth=1)

        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()

    def plot_scatter_F(self):
        if self.F_truth is None or self.F_learned is None:
            print("No F truth/learned. Skipping F scatter.")
            return

        k = self.F_learned.shape[1]
        out = os.path.join(self.plot_dir, "scatter_F_truth_vs_learned.png")

        fig, axs = plt.subplots(1, k, figsize=(4*k, 4))
        if k == 1:
            axs = [axs]

        for i in range(k):
            x = self.F_truth[:, i]
            y = self.F_learned[:, i]
            axs[i].scatter(x, y, alpha=0.35)
            r = np.corrcoef(x, y)[0, 1]
            axs[i].set_title(f"F Source {i+1}\nr={r:.2f}")
            axs[i].set_xlabel("Truth")
            axs[i].set_ylabel("Learned")

            lo = np.nanmin([x.min(), y.min()])
            hi = np.nanmax([x.max(), y.max()])
            axs[i].plot([lo, hi], [lo, hi], "--", linewidth=1)

        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------
    def Excel_results_creation(self):
        if self.F_learned is None or self.G_learned is None:
            raise ValueError("Need F_learned and G_learned set before saving results.")

        # F CSV
        if self.has_ground_truth():
            F_combined = {}
            for i in range(self.F_learned.shape[1]):
                F_combined[f"Learned_{i+1}"] = self.F_learned[:, i]
                F_combined[f"Truth_{i+1}"] = self.F_truth[:, i]
            F_df = pd.DataFrame(F_combined, index=self.get_mz_labels())
        else:
            F_df = pd.DataFrame(self.F_learned, index=self.get_mz_labels(),
                                columns=[f"Learned_{i+1}" for i in range(self.F_learned.shape[1])])
        F_df.to_csv(self.output_F)

        # G CSV
        if self.has_ground_truth():
            G_combined = {}
            for i in range(self.G_learned.shape[1]):
                G_combined[f"Learned_{i+1}"] = self.G_learned[:, i]
                G_combined[f"Truth_{i+1}"] = self.G_truth[:, i]
            G_df = pd.DataFrame(G_combined, index=self.get_time())
        else:
            G_df = pd.DataFrame(self.G_learned, index=self.get_time(),
                                columns=[f"Learned_{i+1}" for i in range(self.G_learned.shape[1])])
        G_df.to_csv(self.output_G)

        # RMSE
        if self.has_ground_truth():
            mse_F = mean_squared_error(self.F_truth, self.F_learned)
            mse_G = mean_squared_error(self.G_truth, self.G_learned)
            pd.DataFrame({"MSE_F": [mse_F], "MSE_G": [mse_G]}).to_csv(self.output_RMSE, index=False)
            print("Saved MSE file.")
            
    def compute_rss(self, X_hat):
        """Compute weighted residual sum of squares."""
        X = self.get_X()
        E = self.get_error()
        return np.mean(((X - X_hat) / E) ** 2)

    # ---------------------------------------------------------
    # Getters / setters (keep your existing API)
    # ---------------------------------------------------------
    def set_F(self, F): self.F_learned = F
    def get_F(self): return self.F_learned
    def set_G(self, G): self.G_learned = G
    def get_G(self): return self.G_learned

    def get_X(self): return self.X
    def get_time(self): return self.time
    def get_mz_labels(self): return self.mz_labels

    def get_F_truth(self): return self.F_truth
    def get_G_truth(self): return self.G_truth

    def get_F_fixed(self): return self.F_fixed
    def get_n_fixed(self): return self.n_fixed

    def get_error(self): return self.E
