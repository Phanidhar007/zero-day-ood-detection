"""End-to-end pipeline: data -> train IDS on KNOWN classes -> score the test
set (benign + known attacks + held-out UNKNOWN) -> OOD AUROC / FPR /
histograms -> metrics.md, metrics.json, per-sample scores and figures.

Run:  python scripts/run_pipeline.py
Kept small on purpose (synthetic data, tiny MLP, few epochs) so it completes
in under ~2-3 minutes on CPU.
"""

from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
from sklearn.preprocessing import StandardScaler

from zood.data import (
    make_splits,
    N_FEATURES,
    N_KNOWN,
    CLASS_NAMES,
    UNKNOWN_CLASS_NAME,
)
from zood.model import IDSMLP, train_model, predict_logits, get_embeddings
from zood.ood import energy_ood_score, fit_mahalanobis_reference, mahalanobis_ood_scores
from zood.evaluate import (
    evaluate_classifier,
    evaluate_ood,
    roc_curves,
    plot_ood_histogram,
    plot_auroc_curve,
    plot_fpr_chart,
    plot_embeddings_scatter,
    write_metrics,
    TPR_TARGET,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_MD = os.path.join(RESULTS_DIR, "metrics.md")
METRICS_JSON = os.path.join(RESULTS_DIR, "metrics.json")
SAMPLES_NPZ = os.path.join(RESULTS_DIR, "ood_scores.npz")

# ---------------------------------------------------------------------------
# Hyper-parameters (small on purpose for fast CPU runs)
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 5000
TEST_FRAC = 0.35
EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3
SHRINK = 0.1


def main() -> None:
    t0 = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("Zero-Day-Style OOD Detection")
    print("=" * 70)

    print("\n[1/4] Building synthetic traffic dataset "
          f"(known: {CLASS_NAMES[:N_KNOWN]}; "
          f"held-out unknown: {UNKNOWN_CLASS_NAME}) ...")
    X_train, y_train, X_test, y_test, groups = make_splits(
        n_samples=N_SAMPLES, test_frac=TEST_FRAC, random_state=SEED,
    )
    n_unk = int((groups == 2).sum())
    n_ben = int((groups == 0).sum())
    n_kn = int((groups == 1).sum())
    print(f"  train (known only): {X_train.shape}  "
          f"test: {X_test.shape} "
          f"[benign={n_ben} known={n_kn} unknown={n_unk}]")

    scaler = StandardScaler().fit(X_train)
    Xs_tr = scaler.transform(X_train).astype(np.float32)
    Xs_te = scaler.transform(X_test).astype(np.float32)

    print("\n[2/4] Training IDS MLP on known classes only ...")
    model = IDSMLP(n_features=N_FEATURES, n_classes=N_KNOWN, seed=SEED)
    train_model(model, Xs_tr, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                lr=LR, seed=SEED, verbose=True)

    print("\n[3/4] Scoring test set with energy + Mahalanobis OOD scorers ...")
    logits_test = predict_logits(model, Xs_te)
    emb_train = get_embeddings(model, Xs_tr)
    emb_test = get_embeddings(model, Xs_te)

    energy_raw = energy_ood_score(logits_test)
    means, cov_inv, cov = fit_mahalanobis_reference(
        emb_train, y_train, n_classes=N_KNOWN, shrink=SHRINK,
    )
    maha = mahalanobis_ood_scores(emb_test, means, cov_inv)

    clf_metrics = evaluate_classifier(model, Xs_te, y_test, groups)
    eval_out = evaluate_ood(energy_raw, maha, groups, tpr_target=TPR_TARGET)

    print(f"  classifier (known classes) acc={clf_metrics['acc']*100:.1f}% "
          f"f1={clf_metrics['f1']:.4f}")
    for key, res in eval_out.items():
        print(f"  {key:>12}: AUROC={res['auroc']:.4f}  "
              f"FPR@benign={res['benign_fpr']:.4f}  "
              f"FPR@known={res['known_fpr']:.4f}  "
              f"sep={res['separation_std']:.2f}std")

    print("\n[4/4] Saving figures, per-sample scores and metrics ...")
    plot_ood_histogram(eval_out, groups, os.path.join(FIGURES_DIR, "ood_histogram.png"))
    plot_auroc_curve(roc_curves(eval_out, groups), eval_out,
                     os.path.join(FIGURES_DIR, "auroc_curve.png"))
    plot_fpr_chart(eval_out, os.path.join(FIGURES_DIR, "fpr_chart.png"))
    plot_embeddings_scatter(emb_test, groups,
                            os.path.join(FIGURES_DIR, "embeddings_scatter.png"))

    idx = np.arange(len(X_test))
    np.savez(
        SAMPLES_NPZ,
        idx=idx,
        class_id=y_test,
        group=groups,
        class_name=np.array([CLASS_NAMES[c] for c in y_test]),
        energy=eval_out["energy"]["scores"],
        mahalanobis=eval_out["mahalanobis"]["scores"],
    )

    setup = dict(
        n_samples=N_SAMPLES, n_features=N_FEATURES, n_known=N_KNOWN,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, shrink=SHRINK,
        test_frac=TEST_FRAC,
    )
    write_metrics(
        setup, clf_metrics, eval_out,
        SAMPLES_NPZ, METRICS_MD, METRICS_JSON, FIGURES_DIR,
    )
    print(f"  metrics -> {METRICS_MD} / {METRICS_JSON}")
    print(f"  samples -> {SAMPLES_NPZ}")
    print(f"  figures -> {FIGURES_DIR}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    print("\n--- KEY METRICS ---")
    print(f"Classifier (known classes): acc={clf_metrics['acc']:.4f} "
          f"f1={clf_metrics['f1']:.4f}")
    for key, res in eval_out.items():
        print(f"{key:>12} | AUROC={res['auroc']:.4f} | "
              f"threshold@{res['tpr_target']*100:.0f}%TPR={res['threshold']:.3f} | "
              f"FPR benign={res['benign_fpr']:.4f} | "
              f"FPR known={res['known_fpr']:.4f} | "
              f"mean known={res['mean_known']:.3f} | "
              f"mean unknown={res['mean_unknown']:.3f} | "
              f"separation={res['separation_std']:.2f} std")


if __name__ == "__main__":
    main()
