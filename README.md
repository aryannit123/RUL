# Remaining Useful Life (RUL) Prediction

A deep-learning project for estimating the **Remaining Useful Life (RUL)** of turbofan engines from multivariate, time-series sensor measurements. The project processes engine operating histories into fixed-length sequences, trains recurrent neural networks, and evaluates predictions on the FD001–FD004 subsets.

Two model approaches are included:

- **LSTM baseline** — a stacked LSTM network.
- **GRU with Attention** — a stacked GRU network followed by a custom temporal attention layer.

The notebooks report RMSE for each dataset and plot predicted RUL against the provided ground-truth RUL.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Contents](#repository-contents)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Implementation Notes](#implementation-notes)

---

## Project Overview

Remaining Useful Life is the estimated number of operating cycles an asset can continue running before it reaches its end of useful life. Estimating RUL from sensor histories can support predictive maintenance by helping identify when equipment may need inspection or replacement.

This project treats RUL estimation as a **supervised time-series regression problem**:

1. Read the engine training, test, and RUL files.
2. Prepare operating-condition and sensor features.
3. Calculate training RUL labels.
4. Scale the input features.
5. Create fixed-length sequences for each engine.
6. Train recurrent neural networks to predict RUL.
7. Predict the RUL of each test engine using its latest observed sequence.
8. Evaluate predictions using Root Mean Squared Error (RMSE).

## Repository Contents

| File | Description |
|---|---|
| `RUL_(1).ipynb` | LSTM-based RUL prediction baseline evaluated on FD001, FD002, FD003, and FD004. |
| `Copy_of_RUL.ipynb` | Includes an earlier FD003-only LSTM experiment and a GRU + Attention implementation evaluated across all four FD subsets. |
| `README.md` | Project documentation. |

## Dataset

The notebooks expect the `CMaps` data directory extracted from a ZIP archive. The code refers to the following files:

```text
CMaps/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

Each training and test data row contains:

- Engine/unit identifier
- Operating cycle (time)
- Three operating-condition settings (`op1`, `op2`, `op3`)
- 21 sensor measurements (`sensor_1` through `sensor_21`)

The model pipelines use **24 input features**: the operating cycle, three operating settings, and 21 sensor measurements. The unit identifier is used to group observations by engine; it is not used as a model input.

The `RUL_FD00X.txt` files provide the reference RUL values used to evaluate the corresponding test subset.

> **Data location:** The notebooks currently expect the archive at `/content/archive (1).zip` and extract it to `/content/archive2`. Update these paths if your archive is stored elsewhere or if you are running outside Google Colab.

## Workflow

### 1. RUL label generation

For each training engine, the target is calculated as:

```python
RUL = maximum_cycle_for_engine - current_cycle
```

This assigns a decreasing RUL target to each observation in the engine's operating history.

### 2. Feature scaling

`StandardScaler` is fitted on the training features. The fitted scaler is then used to transform the corresponding test features.

### 3. Sequence generation

The main FD001–FD004 experiments use:

| Parameter | Value |
|---|---:|
| Sequence/window length | 30 time steps |
| Input features | 24 |
| Training/validation split | 80% / 20% |
| Validation split random state | 83 |

Training sequences are generated with a sliding window within each engine's time-ordered observations. For testing, the latest 30 observations from each engine are used. When a test engine has fewer than 30 observations, the implementation pads the beginning by repeating its first observed row.

### 4. Training

Both main model implementations use:

- Mean Squared Error (MSE) as the training loss
- Adam optimizer
- Initial learning rate of `0.001`
- Batch size of `128`
- Maximum of `100` epochs
- Early stopping based on validation loss, with a patience of `10` epochs and restoration of the best weights

### 5. Evaluation

Predictions are compared against the reference values in each `RUL_FD00X.txt` file. The notebooks calculate Root Mean Squared Error:

\[
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

A lower RMSE represents a smaller prediction error.

The notebooks also plot the true and predicted RUL values for the test engines.

## Models

### 1. Stacked LSTM

Implemented in `RUL_(1).ipynb`.

The network processes a sequence of 30 time steps, each containing 24 features.

| Layer | Configuration |
|---|---|
| LSTM 1 | 128 units, `tanh`, returns sequences |
| LSTM 2 | 64 units, `tanh`, returns sequences |
| LSTM 3 | 32 units, `tanh` |
| Dense 1 | 96 units, `ReLU` |
| Dense 2 | 128 units, `ReLU` |
| Output | 1 unit (predicted RUL) |

**Compilation:** Adam optimizer (`learning_rate=0.001`) with MSE loss.

### 2. GRU with Temporal Attention

Implemented in the later section of `Copy_of_RUL.ipynb`.

| Component | Configuration |
|---|---|
| Input | 30 time steps × 24 features |
| GRU 1 | 128 units, `tanh`, returns sequences |
| GRU 2 | 64 units, `tanh`, returns sequences |
| Attention | Custom trainable attention layer |
| Dense 1 | 96 units, `ReLU` |
| Dense 2 | 128 units, `ReLU` |
| Output | 1 unit (predicted RUL) |

The custom attention layer learns trainable weights to calculate attention scores across the sequence. It applies softmax over the time dimension and combines the GRU outputs into a context vector. The context vector is passed to the dense layers for final RUL regression.

**Compilation:** Adam optimizer (`learning_rate=0.001`) with MSE loss.

## Results

The following values are the RMSE outputs saved in the notebooks. They are presented as recorded and have not been independently reproduced as part of this documentation.

| Dataset | LSTM RMSE (`RUL_(1).ipynb`) | GRU + Attention RMSE (`Copy_of_RUL.ipynb`) |
|---|---:|---:|
| FD001 | 33.69 | 36.59 |
| FD002 | 31.03 | 30.05 |
| FD003 | 47.15 | 35.78 |
| FD004 | 44.52 | 41.41 |

### Result observations

- The LSTM baseline has a lower recorded RMSE on FD001.
- The GRU + Attention model has lower recorded RMSE on FD002, FD003, and FD004.
- The best recorded result in this comparison is **30.05 RMSE** for GRU + Attention on FD002.
- Performance varies across the four subsets, so neither model is best on every dataset.

The notebooks also contain an earlier FD003-only LSTM experiment. Its separate runs show different RMSE values; those results are not mixed into the four-dataset comparison above.

## Requirements

The notebooks use Python and the following libraries:

- NumPy
- Pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib

`zipfile` and `os` are also used from the Python standard library.

Install the main dependencies with:

```bash
python -m pip install numpy pandas scikit-learn tensorflow matplotlib
```

## How to Run

### Option A: Google Colab

1. Upload or open the notebook in Google Colab.
2. Upload the dataset archive using the filename expected by the notebook (`archive (1).zip`), or update the archive path in the first data-loading cell.
3. Confirm that the archive contains the `CMaps` directory and the required FD001–FD004 train, test, and RUL files.
4. Run the notebook cells from top to bottom.
5. Review the printed RMSE values and the true-versus-predicted RUL plots.

### Option B: Local Jupyter Notebook

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Place the dataset archive in your working directory.
4. Update the ZIP archive path and extraction directory in the first notebook cell to valid local paths.
5. Make sure the `data_path` and file paths point to the extracted `CMaps` directory.
6. Open either notebook in Jupyter Notebook, JupyterLab, or VS Code.
7. Run the cells from top to bottom.

Example:

```bash
git clone https://github.com/aryannit123/RUL.git
cd RUL

python -m pip install numpy pandas scikit-learn tensorflow matplotlib
```

Then open the notebook in your preferred Jupyter environment.

## Implementation Notes

- The four-dataset LSTM and GRU + Attention experiments train a separate model for each FD subset.
- Training labels are calculated from each engine's maximum observed training cycle.
- The test predictions use the latest available sequence for each test engine.
- The notebooks contain saved output snapshots. Re-running training may produce different RMSE values because model initialization and other random operations are not fully seeded in every experiment.
- The notebooks are research/experimentation workflows; this repository does not currently include a separate deployment API or production inference service.
