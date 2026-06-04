"""Train and evaluate the login-anomaly classifier in PyTorch.

Run: `python -m pytorch.train` (from the repo root). Exits non-zero if the model
fails to clear sane quality thresholds, so it doubles as a smoke test in CI/Docker.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

from common.data import make_dataset, standardize, train_test_split
from common.metrics import binary_metrics, roc_auc
from pytorch.model import LoginAnomalyNet

EPOCHS = 60
BATCH = 256
LR = 1e-3
SEED = 42


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y = make_dataset(seed=SEED)
    Xtr, ytr, Xte, yte = train_test_split(X, y, test_frac=0.2, seed=1)
    Xtr, Xte = standardize(Xtr, Xte)
    print(f"train={len(Xtr)} test={len(Xte)} positives={int(y.sum())}/{len(y)} "
          f"({y.mean():.1%} anomalous)")

    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte)

    model = LoginAnomalyNet(in_features=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Class weighting: anomalies are the minority, so weight the positive class.
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = len(Xtr_t)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 15 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={total / n:.4f}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).numpy()

    m = binary_metrics(yte, probs)
    auc = roc_auc(yte, probs)
    print("\n=== test metrics (PyTorch) ===")
    print(f"accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  roc_auc={auc:.3f}")
    print(f"confusion: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")

    torch.save(model.state_dict(), "model_pytorch.pt")

    ok = m["accuracy"] >= 0.85 and auc >= 0.85
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
