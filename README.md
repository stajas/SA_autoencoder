# SA_autoencoder
An autoencoder to perform Organic Aerosol Source apportionment from Aerosol Mass Spectrometry measurements using literature derived reference mass spectra

# Fixed-Profile Source-Based Autoencoder

This repository contains a **minimal Source-Based Autoencoder (SourceAE)** pipeline for source apportionment using **known (fixed) profiles**.
The model trains only the encoder while keeping the source profiles fixed.

---

## Overview

The workflow:

1. Load measurements (`Measurement`)
2. Load known source profiles (`F_fixed`)
3. Initialize encoder from the known profiles
4. Train encoder only
5. Export results and diagnostic plots via `Measurement`

The reconstruction follows:

```
X_hat = relu(X @ W^T) @ F^T
```

where:

* **F** — fixed known profiles (not trained)
* **W** — encoder weights (trained)
* **G = relu(X @ W^T)** — source contributions

---



## How to Run

From the **project root**, run:

```bash
python train_sourceae_fixed.py \
  --input "INPUT-PATH" \
  --output /results \
  --fixed_profiles "KNOWN PROFILES PATH" \
  --fixed_labels Names of the sources ex(HOA BBOA "less oxidized" "more oxidized") \
  --epochs 1000 \
  --lr 1e-2
```

---

## Command-Line Arguments

### Required

| Argument           | Description                                   |
| ------------------ | --------------------------------------------- |
| `--input`          | Path to measurement data (.xlsx or .csv)      |
| `--output`         | Output prefix/directory used by `Measurement` |
| `--fixed_profiles` | Excel file containing known source profiles   |

### Optional

| Argument                  | Default       | Description                           |
| ------------------------- | ------------- | ------------------------------------- |
| `--fixed_labels`          | HOA CCOA BBOA | Names of profiles to load             |
| `--epochs`                | 500           | Number of training epochs             |
| `--lr`                    | 1e-2          | Learning rate                         |
| `--random_fixed_profiles` | False         | Randomly select profiles from library |

---

## Outputs

After training, the pipeline automatically generates:

### Excel

* Estimated **F** (profiles)
* Estimated **G** (contributions)
* Additional diagnostics

### Plots

* Source profiles
* Time series of contributions
* Scaled residuals
* Uncertainty visualization
* Ground truth comparison (if available)

All outputs are written using the `Measurement` utilities.

---

## Notes

* Only **fixed-profile SourceAE** is implemented.
* No free-profile learning is performed.
* The encoder is initialized analytically from the known profiles.
* The model enforces non-negative contributions via ReLU.

---

---
**Excel errors**

Install:

```bash
pip install openpyxl
```

**NaN loss**

Try:

* lowering learning rate (`--lr 1e-3`)
* reducing epochs
* checking input uncertainties
