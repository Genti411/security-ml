"""Train and evaluate the SAME login-anomaly classifier in TensorFlow/Keras.

Run: `python -m tensorflow_impl.train` inside the container (the package is named
`tensorflow_impl` there to avoid clashing with the `tensorflow` library import).
Exits non-zero if quality thresholds aren't met, so it doubles as a smoke test.
"""
import sys

import numpy as np
import tensorflow as tf

from common.data import make_dataset, standardize, train_test_split
from common.metrics import binary_metrics, roc_auc

EPOCHS = 60
BATCH = 256
SEED = 42


def main() -> int:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    X, y = make_dataset(seed=SEED)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.2, seed=1)
    Xtr, Xte = standardize(Xtr, Xte)
    print(f"train={len(Xtr)} test={len(Xte)} positives={int(y.sum())}/{len(y)} "
          f"({y.mean():.1%} anomalous)")

    model = tf.keras.Sequential([
        tf.keras.layers.Input((X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])

    pos = int(ytr.sum())
    neg = len(ytr) - pos
    class_weight = {0: 1.0, 1: neg / max(1, pos)}  # up-weight the minority class
    model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, verbose=0,
              class_weight=class_weight)

    probs = model.predict(Xte, verbose=0).ravel()
    m = binary_metrics(yte, probs)
    auc = roc_auc(yte, probs)
    print("\n=== test metrics (TensorFlow/Keras) ===")
    print(f"accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  roc_auc={auc:.3f}")
    print(f"confusion: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")

    model.save("model_tf.keras")

    ok = m["accuracy"] >= 0.85 and auc >= 0.85
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
