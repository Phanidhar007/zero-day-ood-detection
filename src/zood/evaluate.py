"""Evaluation: OOD AUROC, FPR@threshold, histograms, plots, metrics writer.

This module ties the pieces together:

- evaluates the IDS classifier on the in-distribution (known-class) test set,
- computes energy + Mahalanobis OOD scores over the test set (which contains
  benign, known attacks and the held-out unknown attack),
- measures **OOD AUROC** (unknown vs known), the **threshold** that recalls
  95% of the unknown class, and the **FPR on normal traffic** (and on known
  attacks) at that threshold,
- saves figures to ``results/figures/`` and writes ``metrics.md`` / JSON.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.decomposition import PCA

from .data import CLASS_NAMES, KNOWN_CLASS_NAMES, UNKNOWN_CLASS_NAME
from .model import predict_logits, predict_proba, get_embeddings

# ---------------------------------------------------------------------------
# AI Shield dark theme used for all saved figures
# ---------------------------------------------------------------------------
DARK_BG = "#09090b"
CARD = "#131316"
EMERALD = "#10b981"
EMERALD_LT = "#34d399"
RED = "#ef4444"
PURPLE = "#c084fc"
CYAN = "#06b6d2"
AMBER = "#fbbf24"
TEXT = "#e4e4e7"

GROUPS = {0: "benign", 1: "known attack", 2: "unknown (zero-day)"}

# default detection operating point: recall this fraction of the unknown class
TPR_TARGET = 0.95


def _style_ax(ax, title, ylabel, xlabel=""):
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT)
    ax.grid(True, color="#1f1f23", linewidth=0.6)
    for s in ax.spines.values():
        s.set_color("#27272a")


def _init_figure(figsize=(7.5, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    return fig, ax


# ---------------------------------------------------------------------------
# Classifier (in-distribution) evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(model, X_test, y_test, groups):
    """Accuracy / F1 on known-class test samples + per-class breakdown."""
    logits = predict_logits(model, X_test)
    pred = logits.argmax(axis=1)

    known = groups != 2
    acc = float(accuracy_score(y_test[known], pred[known]))
    f1 = float(f1_score(y_test[known], pred[known], average="macro"))
    per_class = {}
    for c, name in enumerate(KNOWN_CLASS_NAMES):
        mask = y_test == c
        per_class[name] = float(accuracy_score(y_test[mask], pred[mask]))
    return {"acc": acc, "f1": f1, "per_class": per_class}


# ---------------------------------------------------------------------------
# OOD evaluation
# ---------------------------------------------------------------------------


def _orient_scores(raw, unknown_mask, known_mask):
    """Ensure higher score = more OOD (calibrate the sign empirically)."""
    pos = unknown_mask
    neg = known_mask
    if pos.sum() == 0 or neg.sum() == 0:
        return raw, 0.5
    auc = float(roc_auc_score(unknown_mask, raw))
    if auc < 0.5:
        raw = -raw
        auc = 1.0 - auc
    return raw, auc


def evaluate_ood(energy_raw, maha, groups, tpr_target=TPR_TARGET):
    """Compute per-scorer OOD metrics (AUROC, threshold@TPR, FPRs, separation).

    ``groups``: 0 = benign, 1 = known attack, 2 = unknown.
    """
    unknown = groups == 2
    known_all = ~unknown  # benign + known attacks (the "known" population)
    benign = groups == 0
    known_attack = groups == 1

    out = {}
    for key, raw in (("energy", energy_raw), ("mahalanobis", maha)):
        scores, auc = _orient_scores(raw.copy(), unknown, known_all)
        unk_scores = scores[unknown]
        threshold = float(np.quantile(unk_scores, tpr_target))
        benign_fpr = float(np.mean(scores[benign] >= threshold))
        known_fpr = float(np.mean(scores[known_attack] >= threshold))
        mu_known = float(np.mean(scores[known_all]))
        mu_unknown = float(np.mean(scores[unknown]))
        std_known = float(np.std(scores[known_all]))
        out[key] = {
            "auroc": auc,
            "tpr_target": float(tpr_target),
            "threshold": threshold,
            "benign_fpr": benign_fpr,
            "known_fpr": known_fpr,
            "mean_known": mu_known,
            "mean_unknown": mu_unknown,
            "separation": mu_unknown - mu_known,
            "separation_std": (mu_unknown - mu_known) / max(std_known, 1e-9),
            "scores": scores,
        }
    return out


def roc_curves(eval_out, groups):
    """Return {(scorer): (fpr, tpr)} for plotting."""
    unknown = groups == 2
    curves = {}
    for key, res in eval_out.items():
        fpr, tpr, _ = roc_curve(unknown, res["scores"])
        curves[key] = (fpr, tpr)
    return curves


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_ood_histogram(eval_out, groups, path):
    """Known vs benign vs unknown score distributions for both scorers."""
    names = {"energy": "Energy OOD score", "mahalanobis": "Mahalanobis distance"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(DARK_BG)
    for ax, (key, res) in zip(axes, eval_out.items()):
        s = res["scores"]
        bins = np.linspace(min(s.min(), 0), s.max(), 60)
        ax.hist(s[groups == 0], bins=bins, alpha=0.45, color=CYAN,
                label="benign")
        ax.hist(s[groups == 1], bins=bins, alpha=0.45, color=EMERALD,
                label="known attack")
        ax.hist(s[groups == 2], bins=bins, alpha=0.55, color=RED,
                label="unknown (zero-day)")
        ax.axvline(res["threshold"], color=AMBER, ls="--", lw=1.4,
                   label=f"threshold@{res['tpr_target']:.0%}")
        ax.legend(facecolor=CARD, edgecolor="#27272a", labelcolor=TEXT,
                  fontsize=8, loc="upper right")
        _style_ax(ax, f"{names[key]} (AUROC={res['auroc']:.3f})",
                  "count")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=DARK_BG)
    plt.close(fig)


def plot_auroc_curve(curves, eval_out, path):
    """ROC curves of the OOD detectors (unknown vs known)."""
    fig, ax = _init_figure()
    ax.plot([0, 1], [0, 1], ls="--", color="#52525b", lw=1, label="chance")
    styles = {"energy": (EMERALD, "-o"), "mahalanobis": (PURPLE, "-s")}
    for key, (fpr, tpr) in curves.items():
        color, marker = styles[key]
        ax.plot(fpr, tpr, marker, color=color, lw=2, ms=3, alpha=0.9,
                label=f"{key} (AUROC={eval_out[key]['auroc']:.3f})")
    ax.legend(facecolor=CARD, edgecolor="#27272a", labelcolor=TEXT,
              loc="lower right")
    _style_ax(ax, "OOD detection ROC: unknown attack vs known traffic",
              "TPR (unknown detected)", "FPR")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=DARK_BG)
    plt.close(fig)


def plot_fpr_chart(eval_out, path):
    """FPR on benign and on known attacks at the chosen TPR operating point."""
    fig, ax = _init_figure((7.5, 4.2))
    keys = list(eval_out.keys())
    x = np.arange(len(keys))
    w = 0.32
    benign_fpr = [eval_out[k]["benign_fpr"] for k in keys]
    known_fpr = [eval_out[k]["known_fpr"] for k in keys]
    b1 = ax.bar(x - w / 2, benign_fpr, w, color=CYAN, label="FPR on benign")
    b2 = ax.bar(x + w / 2, known_fpr, w, color=AMBER, label="FPR on known attacks")
    for bars, vals in ((b1, benign_fpr), (b2, known_fpr)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.4f}", ha="center", color=TEXT, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(@{eval_out[k]['tpr_target']:.0%} TPR on unknown)"
                        for k in keys])
    ax.set_ylim(0, max(0.1, max(benign_fpr + known_fpr) * 1.15))
    ax.legend(facecolor=CARD, edgecolor="#27272a", labelcolor=TEXT)
    _style_ax(ax, "False-positive rate at the detection threshold", "FPR")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=DARK_BG)
    plt.close(fig)


def plot_embeddings_scatter(embeddings, groups, path, n_max=1500):
    """PCA(2) of the penultimate-layer embeddings, coloured by group."""
    rng = np.random.RandomState(0)
    if len(embeddings) > n_max:
        idx = rng.choice(len(embeddings), n_max, replace=False)
        emb, grp = embeddings[idx], groups[idx]
    else:
        emb, grp = embeddings, groups
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(emb)
    fig, ax = _init_figure((7.5, 5))
    colors = {0: CYAN, 1: EMERALD, 2: RED}
    for g, name in GROUPS.items():
        mask = grp == g
        ax.scatter(z[mask, 0], z[mask, 1], s=14, alpha=0.65,
                   color=colors[g], label=name)
    ax.legend(facecolor=CARD, edgecolor="#27272a", labelcolor=TEXT)
    _style_ax(ax, "Penultimate-layer embeddings (PCA, 2-d)",
              "PC2", "PC1")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=DARK_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics writer
# ---------------------------------------------------------------------------


def write_metrics(
    setup: dict,
    classifier_metrics: dict,
    eval_out: dict,
    scores_npz_path: str,
    metrics_md_path: str,
    metrics_json_path: str,
    figures_dir: str,
) -> None:
    """Write results/metrics.md + metrics.json + the per-sample score table."""
    energy = eval_out["energy"]
    maha = eval_out["mahalanobis"]
    per_class = classifier_metrics["per_class"]
    pc_row = " | ".join(f"{n}: {per_class[n]*100:.1f}%" for n in KNOWN_CLASS_NAMES)

    table = (
        "| scorer | OOD AUROC | threshold @95% TPR | FPR on benign | "
        "FPR on known attacks | mean known | mean unknown | separation (std) |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| energy | {energy['auroc']:.4f} | {energy['threshold']:.3f} "
        f"| {energy['benign_fpr']:.4f} | {energy['known_fpr']:.4f} "
        f"| {energy['mean_known']:.3f} | {energy['mean_unknown']:.3f} "
        f"| {energy['separation_std']:.2f} |\n"
        f"| mahalanobis | {maha['auroc']:.4f} | {maha['threshold']:.3f} "
        f"| {maha['benign_fpr']:.4f} | {maha['known_fpr']:.4f} "
        f"| {maha['mean_known']:.3f} | {maha['mean_unknown']:.3f} "
        f"| {maha['separation_std']:.2f} |\n"
    )

    md = f"""# Zero-Day-Style OOD Detection - results (real run)

Generated by `scripts/run_pipeline.py` on this machine. All numbers below are
from an actual local run (not fabricated).

## Setup

- synthetic network-flow dataset: `{setup['n_samples']}` records,
  `{setup['n_features']}` features, `{len(CLASS_NAMES)}` classes
- known classes trained on: {", ".join(KNOWN_CLASS_NAMES)}
- held-out unknown (zero-day proxy): **{UNKNOWN_CLASS_NAME}** - never seen
  during training, present only in the test set
- IDS model: torch MLP ({setup['n_features']} -> 64 -> 32(embedding) -> {setup['n_known']}),
  trained on known classes for {setup['epochs']} epochs (Adam, lr={setup['lr']})
- OOD scorers (manual, pytorch-ood NOT used):
  - **energy**: `-E(x) = T*logsumexp(logits/T)` over the class logits,
  - **Mahalanobis**: min per-class Mahalanobis distance to the known-class
    Gaussian fit (pooled covariance, shrinkage={setup['shrink']}) on the
    32-d penultimate-layer embeddings.
- operating point: threshold that recalls `{TPR_TARGET*100:.0f}%` of the
  unknown class; FPR measured on benign traffic and on known attacks.
- score direction calibrated so **higher score => more OOD-like**.

## Classifier (in-distribution)

- known-class test accuracy: `{classifier_metrics['acc']*100:.1f}%`
- macro F1 (known classes): `{classifier_metrics['f1']:.4f}`
- per-class accuracy: {pc_row}

## OOD detection results

{table}

## Figures

- `figures/ood_histogram.png` - known vs benign vs unknown score histograms
- `figures/auroc_curve.png` - OOD detection ROC (both scorers)
- `figures/fpr_chart.png` - FPR on benign / known attacks at the threshold
- `figures/embeddings_scatter.png` - PCA(2-d) of embeddings, by group

## Reading the table

- **OOD AUROC**: probability that a random unknown sample scores more OOD-like
  than a random known sample. ~1.0 = near-perfect separation of the held-out
  zero-day attack.
- **FPR on benign** at the 95%-recall threshold: how often normal traffic is
  wrongly flagged as OOD - the operational cost of the detector.
- **FPR on known attacks**: known attacks wrongly flagged as OOD (they are
  *known*, so ideally ~0).
- **separation (std)**: effect size `(mean_unknown - mean_known) / std_known`
  of the score distributions.
"""
    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(md)

    json_payload = {
        "setup": setup,
        "classifier": classifier_metrics,
        "tpr_target": TPR_TARGET,
        "energy": {k: v for k, v in energy.items() if k != "scores"},
        "mahalanobis": {k: v for k, v in maha.items() if k != "scores"},
        "samples_npz": os.path.basename(scores_npz_path),
        "figures": sorted(os.listdir(figures_dir)),
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, default=float)
    return md
