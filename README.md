# Security ML — Login-Anomaly Classifier (PyTorch + TensorFlow)

A supervised neural-network classifier that flags **anomalous (account-takeover-
like) logins** from behavioral features, implemented **twice** — once in
**PyTorch**, once in **TensorFlow/Keras** — against the same data and metrics so
the two are directly comparable. This extends the scikit-learn login-anomaly work
into real deep-learning frameworks.

| Area | What's shown |
|------|--------------|
| **PyTorch** | Custom `nn.Module` MLP, `BCEWithLogitsLoss` with class weighting, training loop, evaluation |
| **TensorFlow / Keras** | Equivalent `Sequential` model, `class_weight`, the same task end-to-end |
| **ML engineering** | train/test split with **no leakage** (standardize on train stats only), class-imbalance handling, ROC-AUC + precision/recall (not just accuracy), reproducible synthetic data |
| **Security framing** | features mirror a real detection pipeline (impossible-travel velocity, new-device, geo risk, failed attempts, off-hours) |

The dataset is generated deterministically (`common/data.py`) — no download — so
results reproduce exactly.

## Run

```bash
# PyTorch
docker build -f pytorch/Dockerfile -t security-ml-pytorch .
docker run --rm security-ml-pytorch

# TensorFlow
docker build -f tensorflow/Dockerfile -t security-ml-tensorflow .
docker run --rm security-ml-tensorflow
```

Each prints test-set metrics and a `RESULT: PASS/FAIL` (it fails the run if
accuracy or ROC-AUC drops below 0.85, so the training scripts double as CI smoke
tests).

Locally (with a venv) you can also run `python -m pytorch.train` after
`pip install -r pytorch/requirements.txt`.

## Tests

```bash
python -m pytest tests/      # pure-numpy: dataset, no-leakage standardize, metrics
```

## Why two frameworks

PyTorch and Keras express the same idea differently — an explicit training loop
with `BCEWithLogitsLoss` vs a declarative `model.compile/fit`. Building both on
one task makes the trade-offs concrete (control vs brevity) and shows the model,
not the framework, is what's understood.

## Layout

```
common/data.py       reproducible synthetic login dataset + split/standardize
common/metrics.py    dependency-free accuracy/precision/recall/F1/ROC-AUC
pytorch/             model.py (MLP) + train.py + Dockerfile
tensorflow/          train.py (Keras) + Dockerfile
tests/test_data.py   numpy-only unit tests
```

## Note on the data

The dataset is **synthetic** (clearly labeled as such) so the project is
self-contained and reproducible. The pipeline — features, split, standardization,
class weighting, model, metrics — is exactly what you'd run on real auth logs;
only the data source would change.
