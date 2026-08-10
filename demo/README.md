# Demo

Interactive demo in the AI Shield dark theme. A lightweight **live dashboard** is deployed at https://zero-day-ood-detection.vercel.app (real metrics + figures); this Streamlit app is the full local version.

```bash
python scripts/run_pipeline.py          # first: populate results/metrics.json, ood_scores.npz, figures
streamlit run demo/app.py
```

Shows:
- Stat cards: OOD AUROC (energy + Mahalanobis), FPR on benign traffic at the detection threshold, known-class classifier accuracy
- Saved figures: known-vs-unknown OOD score histogram, OOD detection ROC
- **Interactive OOD score explorer**: pick the scorer (energy / Mahalanobis), drag the desired recall of the unknown class (TPR), and see the live histogram, threshold line, and FPR on benign / known attacks
- Per-sample OOD score table with OOD flags at the selected threshold


## 🌐 Live demo

https://zero-day-ood-detection.vercel.app — real metrics + figures dashboard (AI Shield theme).
