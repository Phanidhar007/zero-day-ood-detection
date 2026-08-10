# Zero-Day-Style OOD Detection

Train an IDS classifier on **known** attack classes, then build a separate **OOD (out-of-distribution) scorer** — energy-based and Mahalanobis distance in the penultimate-layer embedding space — that flags inputs which don't resemble any known-class distribution. A held-out attack class (`data-exfiltration`) plays the role of a zero-day at test time.

## 🌐 Live demo

**https://zero-day-ood-detection.vercel.app** — live results dashboard: real metrics from `results/metrics.md` plus charts from `results/figures/` (AI Shield dark theme, no model executed server-side).

Interactive **local** demo (Streamlit): `streamlit run demo/app.py` — see [demo/README.md](demo/README.md).

## Threat Model

**What we defend against:** a *zero-day style unknown attack* — an intrusion technique that was never present in the training data. A supervised IDS has no label for it, so it will happily classify the malicious traffic into one of the known classes (e.g. call an exfiltration campaign "malware-c2"). The defense is not to detect the specific attack (impossible a priori) but to detect that the input **does not fit any known-class distribution**, triggering "unknown / escalate to analyst" instead of a confident (wrong) label.

**Why it matters:** zero-day attacks are precisely the ones that exploit the gap between what a model was trained on and what an attacker actually ships. A purely supervised classifier is overconfident exactly where it is most dangerous.

**Adversary capability assumptions:** the adversary uses a novel technique whose traffic signature does not match any known attack class (by definition of a zero-day). The defender controls the feature extraction and the training-time attack taxonomy; the OOD scorer must be robust to inputs that sit anywhere in feature space, not just near the decision boundaries.

**How this project demonstrates it:**

1. The IDS classifier is trained **only** on `benign, portscan, brute-force, malware-c2` — `data-exfiltration` is deliberately excluded from training.
2. Two OOD scorers are built **manually** (no `pytorch-ood` dependency):
   - **Energy-based**: `score = -E(x)` with `E(x) = -T·logsumexp(logits/T)` over the class logits.
   - **Mahalanobis**: per-class Gaussian fit (mean + pooled covariance with shrinkage) on the 32-d penultimate-layer embeddings; score = minimum Mahalanobis distance to the known-class centroids.
3. The held-out `data-exfiltration` class is scored at test time and cleanly separated from known traffic (OOD AUROC), with the operational false-positive rate measured on normal (benign) traffic.

## Results (real run)

From `results/metrics.md` (5000 synthetic network-flow samples / 18 features, torch MLP `18→64→32(embedding)→4`, operating point = threshold recalling 95% of the unknown class):

| scorer | OOD AUROC | threshold @95% TPR | FPR on benign | FPR on known attacks | mean known | mean unknown | separation (std) |
|---|---|---|---|---|---|---|---|
| energy | 0.9365 | -0.905 | 0.0000 | 0.0000 | -7.807 | -3.206 | 1.91 |
| mahalanobis | 0.8760 | 7.135 | 0.0071 | 0.0318 | 3.881 | 5.652 | 0.91 |

In-distribution classifier (known classes): **98.6%** test accuracy, macro F1 **0.986** (benign 98.7% / portscan 99.6% / brute-force 98.1% / malware-c2 97.7%).

**Reading the table:** both scorers separate the held-out zero-day attack from known traffic (AUROC ≳ 0.88, ≳0.9σ score separation); at the operating point that catches 95% of the unknown class, normal-traffic false positives are 0.0% (energy) and 0.7% (Mahalanobis). The supervised classifier meanwhile still classifies known attacks at 98.6% accuracy — i.e. it does not sacrifice in-distribution performance to become OOD-aware.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run

```bash
# Full pipeline: build data, train IDS on known classes, score the test set
# (incl. held-out unknown), write results/metrics.md + figures
python scripts/run_pipeline.py

# Local interactive demo (Streamlit, AI Shield dark theme)
streamlit run demo/app.py

# Experiment notebook
jupyter notebook notebooks/ood_experiment.ipynb
```

## Code Layout

```
src/zood/
  data.py     # synthetic network-flow dataset + splits; data-exfiltration held out
  model.py    # small torch MLP IDS classifier with an exposed 32-d embedding layer
  ood.py      # manual energy-based + Mahalanobis OOD scorers (pytorch-ood NOT required)
  evaluate.py # OOD AUROC, FPR@threshold, known-vs-unknown histograms, plots, metrics
scripts/run_pipeline.py   # end-to-end pipeline (train -> score -> evaluate -> metrics)
notebooks/ood_experiment.ipynb
demo/app.py               # local Streamlit demo (AI Shield dark theme)
results/                  # metrics.md (real run) + figures/
```

`pytorch-ood` is listed in `requirements.txt` as an optional drop-in; both OOD scorers are implemented manually in `src/zood/ood.py` so the repo runs end-to-end with only the installed packages.
