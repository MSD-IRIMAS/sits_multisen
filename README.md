# MULTISENGE: Pixel-Based Time Series Processing & Analysis

This repository contains the pipeline to process, structure, and analyze satellite image time series (SITS) from the **MULTISENGE** dataset. It handles the extraction of pixel-level time series, organizes them into machine learning-ready cross-validation folds, calculates global statistics, and performs temporal correlation analyses.

## Data Availability

The code relies on the **MULTISENGE** dataset. You must download the necessary raw data (Sentinel-2 images, Ground Reference, and Labels) from Zenodo:

[**Dataset: MULTISENGE (Zenodo)**](https://zenodo.org/records/6375466)

To run this pipeline, ensure you have the following components extracted locally:
* **Sentinel-2 Images** (`s2_images`)
* **Ground Reference** (GeoTIFF masks)
* **Labels** (JSON files describing the patches)

## 🛠 Installation

To simplify execution, this project uses a `pyproject.toml` configuration.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/multisenge-processing.git](https://github.com/your-username/multisenge-processing.git)
    cd multisenge-processing
    ```

2.  **Install in editable mode:**
    Run the following command to install dependencies and register the `multisenge` executable on your system:
    ```bash
    pip install -e .
    ```

## ⚙️ Configuration (Mandatory)

The project uses **Hydra** for configuration management. All settings are defined in `config/hydra_config.yaml`.

### Variables you MUST update
Before running any command, open `config/hydra_config.yaml` and update the `dataset.paths` section with the actual locations of your downloaded data.

You **must** provide valid paths for these 4 variables:

1.  `folds_csv`: Directory containing your split CSVs (`fold_0.csv`, etc.).
2.  `labels`: Directory containing the JSON label files.
3.  `ground_reference`: Directory containing the GeoTIFF masks.
4.  `s2_images`: Directory containing the Sentinel-2 images.