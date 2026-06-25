# Aerofoil Aerodynamic Coefficients ML Predictive Model

By Niccolò Forte  
Supervised by Dr. Jens-Dominik Mueller

This repository contains the software developed for a Bachelor dissertation submitted to the School of Engineering and
Materials Science at Queen Mary University of London.

The project builds a machine learning workflow for predicting aerofoil aerodynamic coefficients, especially lift
coefficient (`Cl`) and drag coefficient (`Cd`), across aerofoil geometries, angles of attack (`AoA`), and Reynolds
numbers (`Re`). The model is a Keras multi-layer perceptron trained primarily on XFOIL/Airfoil Tools data, with support
for limited experimental and Rennes datasets.

The current codebase supports two ways of working:

- `main.py` for a one-shot command-line training/evaluation/prediction run.
- notebooks in `jupyter/` for iterative exploration while importing the same implementation used by the scripts.

## Current Structure

```text
.
|-- main.py                 # CLI entry point for the complete workflow
|-- resources/              # Canonical implementation modules
|   |-- aerofoils.py         # Aerofoil geometry download, parsing, interpolation, plotting
|   |-- cases.py             # Aerodynamic coefficient download, parsing, saving/loading
|   |-- data.py              # DataFrame merging, train/test split, NN input/output preparation
|   |-- nnetwork.py          # Keras model, metrics, prediction plotting
|   `-- saved.py             # Keras model and result persistence helpers
|-- aerofoils.py             # Backwards-compatible wrapper for resources.aerofoils
|-- cases.py                 # Backwards-compatible wrapper for resources.cases
|-- data.py                  # Backwards-compatible wrapper for resources.data
|-- nnetwork.py              # Backwards-compatible wrapper for resources.nnetwork
|-- saved.py                 # Backwards-compatible wrapper for resources.saved
|-- jupyter/                 # Notebook front-ends that import resources/
|-- dat/                     # Bundled raw aerofoil/case data
|-- dat-saved/               # Preprocessed case DataFrames
|-- scripts/smoke_check.py   # Lightweight repository checks
`-- requirements.txt         # Python dependencies
```

The `resources/` package is now the source of truth. The top-level module files remain so older imports such as
`import saved` still work, but new code should import from `resources`.

## Installation

Use Python 3.11 or 3.12 with the current dependency stack. A virtual environment is recommended.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies are Pandas, NumPy, SciPy, Matplotlib, BeautifulSoup/lxml, scikit-learn, TensorFlow, and Keras.
Keras 3 is used, so saved models are written as `.keras` files.

## Smoke Checks

After installing Python, run the lightweight repository check from the repo root:

```bash
python scripts/smoke_check.py
```

After installing dependencies, you can also verify imports:

```bash
python scripts/smoke_check.py --imports
```

These checks validate repository structure and notebook JSON. They are not a full training test.

## Running The Workflow

Run the default one-shot workflow:

```bash
python main.py
```

Run with explicit training and prediction settings:

```bash
python main.py --epochs 20 --batch 256 --lr 0.001 --n-train 250 --n-test 50 --prediction-name n0012 --prediction-re 1000000
```

Run the Rennes training path:

```bash
python main.py --ren --prediction-name n0012 --prediction-re 1000000
```

Rebuild downloaded/saved case data before training:

```bash
python main.py --reset
```

Useful CLI options:

| Option | Purpose |
| --- | --- |
| `--reset` | Download/rebuild raw and saved case data before running. |
| `--ren` | Train using the Rennes profile/coefficient path and test against the configured experimental target. |
| `--epochs` | Number of training epochs. |
| `--batch` | Keras training batch size. |
| `--lr` | Adam optimizer learning rate. |
| `--n-train` | Number of aerofoils selected for training. |
| `--n-test` | Number of aerofoils selected for testing in the standard path. |
| `--prediction-name` | Aerofoil file/name to plot if present in predictions. |
| `--prediction-re` | Reynolds number to plot if present in predictions. |
| `--verbose` | Keras verbosity level. |
| `--no-callbacks` | Disable training callbacks. |

If the requested prediction target is not present in the current test split, the plotting helper falls back to an
available aerofoil/Re pair and prints a warning.

## Data

The repository currently tracks the `dat/` folder intentionally. It contains 11,034 raw data files, about 49 MB, covering:

- UIUC aerofoil coordinate `.dat` files.
- Airfoil Tools aerofoil coordinate and XFOIL coefficient files.
- Rennes aerofoil/coefficient files.
- Experimental coefficient CSVs digitised or adapted from literature sources.

The `dat-saved/` folder contains preprocessed case DataFrames used by the default workflow. Running with `--reset`
rebuilds these files from raw/downloaded data and overwrites the saved CSVs.

The download helpers depend on third-party sites. If those sites change their HTML or file layout, `--reset` may need
maintenance even if the bundled data still works.

## Outputs

Generated model outputs are written under `models/`, which is ignored by Git.

A typical run creates:

```text
models/
`-- MLP-Sigmoid/
    |-- model.keras
    |-- predictions/
    |   `-- <prediction-set>/
    |       |-- output.csv
    |       `-- pred-metrics.csv
    |-- training/
    |   `-- fitHistory.csv
    `-- evaluation/
        `-- evaluate.csv
```

Models are saved in Keras 3 `.keras` format. Legacy TensorFlow SavedModel directories cannot be loaded with
`keras.models.load_model()` in this dependency stack; retrain and save again to create `model.keras`.

## Notebook Workflow

The notebooks in `jupyter/` are exploration/demo front-ends. They import the same modules used by `main.py`.

Use notebooks when you want to inspect intermediate DataFrames, rerun selected steps, or experiment with plotting/model
settings. Use `main.py` when you want a reproducible one-shot execution.

## Known Limitations

- This is a dissertation-era MLP workflow, not a production aerodynamic solver.
- Predictions depend on the training data distribution and the selected aerofoil/Re/AoA coverage.
- Some raw data is sourced from third-party websites and may become unavailable or change format over time.

## Citation And License

Please cite the software using the metadata in [`CITATION.cff`](CITATION.cff).

This project is released under the MIT License. See [`LICENSE.md`](LICENSE.md).
