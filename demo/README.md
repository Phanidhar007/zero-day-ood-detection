# Demo

Local interactive demo (AI Shield dark theme). NOT deployed — heavy deps (torch MLP + OOD scoring) can't run on serverless.

```bash
python scripts/run_pipeline.py          # first: populate results/metrics.json, ood_scores.npz, figures
streamlit run demo/app.py
```

Shows:
- Stat cards: OOD AUROC (energy + Mahalanobis), FPR on benign traffic at the detection threshold, known-class classifier accuracy
- Saved figures: known-vs-unknown OOD score histogram, OOD detection ROC
- **Interactive OOD score explorer**: pick the scorer (energy / Mahalanobis), drag the desired recall of the unknown class (TPR), and see the live histogram, threshold line, and FPR on benign / known attacks
- Per-sample OOD score table with OOD flags at the selected threshold
