"""Zero-Day-Style OOD Detection -- LOCAL Streamlit demo (AI Shield dark theme).

Run locally (NOT deployed):
    streamlit run demo/app.py

Requires results/metrics.json + results/ood_scores.npz, produced by
`python scripts/run_pipeline.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics.json"
SAMPLES = ROOT / "results" / "ood_scores.npz"

st.set_page_config(page_title="Zero-Day OOD Detection | AI Shield", layout="wide")

CSS = """<style>
  :root { --bg:#030303; --card:#09090b; --card-95:rgba(9,9,11,.95);
    --border:#18181b; --border-strong:#27272a;
    --emerald-400:#34d399; --emerald-500:#10b981;
    --purple-400:#c084fc; --cyan-500:#06b6d2;
    --red-400:#f87171; --red-500:#ef4444; --amber-400:#fbbf24;
    --text:#ffffff; --text-2:#d4d4d8; --text-3:#a1a1aa; --text-4:#71717a; --text-5:#52525b;
    --font-heading:"Space Grotesk", sans-serif;
    --font-body:"Plus Jakarta Sans", sans-serif;
    --font-mono:ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .stApp { background:#030303; color:var(--text); font-family:var(--font-body); }
  h1,h2,h3,h4 { font-family:var(--font-heading); letter-spacing:-0.02em; }
  .grad-title { background:linear-gradient(135deg,#fff,#f4f4f5 50%,#a1a1aa);
    -webkit-background-clip:text; background-clip:text; color:transparent; }
  .grad-accent { background:linear-gradient(90deg,#34d399,#5eead4,#06b6d2);
    -webkit-background-clip:text; background-clip:text; color:transparent; }
  .card { border-radius:16px; border:1px solid var(--border); background:var(--card-95);
    box-shadow:0 20px 50px rgba(0,0,0,.85); padding:20px; margin-bottom:16px; }
  .stat-card { padding:18px 20px; border-radius:16px; border:1px solid var(--border);
    background:var(--card-95); box-shadow:0 8px 30px rgba(0,0,0,.8); margin-bottom:12px; }
  .stat-card .label { font-family:var(--font-mono); font-size:10px; color:var(--text-4);
    letter-spacing:.12em; text-transform:uppercase; margin-bottom:4px; }
  .stat-card .value { font-family:var(--font-mono); font-size:22px; font-weight:800; }
  .badge { font-family:var(--font-mono); font-size:9px; font-weight:700; text-transform:uppercase;
    padding:3px 8px; border-radius:4px; background:rgba(16,185,129,.1); color:var(--emerald-400);
    letter-spacing:.08em; display:inline-block; }
  .badge.danger { background:rgba(239,68,68,.1); color:var(--red-400); }
  .badge.purple { background:rgba(192,132,252,.1); color:var(--purple-400); }
  .badge.warn { background:rgba(245,158,11,.1); color:var(--amber-400); }
  .section-label { font-family:var(--font-mono); font-size:11px; color:var(--emerald-400);
    letter-spacing:.15em; text-transform:uppercase; margin-bottom:6px; display:block; }
  .mono { font-family:var(--font-mono); }
  .muted { color:var(--text-3); }
  .hint { font-family:var(--font-mono); font-size:11px; color:var(--text-4);
    letter-spacing:.08em; text-transform:uppercase; }
  .footer { border-top:1px solid rgba(255,255,255,.05); padding:2rem 0 0;
    text-align:center; color:var(--text-5); font-size:12px; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_metrics():
    if not METRICS.exists():
        return None
    with open(METRICS, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_samples():
    if not SAMPLES.exists():
        return None
    data = np.load(SAMPLES, allow_pickle=True)
    return {
        "idx": data["idx"],
        "class_name": data["class_name"].astype(str),
        "group": data["group"],
        "energy": data["energy"],
        "mahalanobis": data["mahalanobis"],
    }


GROUP_NAMES = {0: "benign", 1: "known attack", 2: "unknown (zero-day)"}
GROUP_COLORS = {0: "#06b6d2", 1: "#10b981", 2: "#ef4444"}

st.markdown('<span class="section-label">AI SHIELD // INTRUSION DETECTION</span>',
            unsafe_allow_html=True)
st.markdown('<h1 class="grad-title">Zero-Day-Style OOD Detection</h1>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="muted" style="font-size:15px">An IDS trained on <span class="grad-accent">known</span> '
    'attack classes gets a second stage that flags anything it was never trained to recognise. '
    '<code>data-exfiltration</code> is held out as a <span class="grad-accent">zero-day proxy</span>.</p>',
    unsafe_allow_html=True)

metrics = load_metrics()
samples = load_samples()
if not metrics or not samples:
    st.markdown('<p class="hint">Run <code>python scripts/run_pipeline.py</code> first to populate '
                'results/metrics.json and results/ood_scores.npz</p>', unsafe_allow_html=True)
    st.stop()

energy = metrics["energy"]
maha = metrics["mahalanobis"]
clf = metrics["classifier"]

# ----------------------------------------------------------------- stat cards
st.markdown('<div class="card"><span class="section-label">MEASURED RESULTS '
            '(from scripts/run_pipeline.py)</span>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    f'<div class="stat-card"><div class="label">OOD AUROC - energy</div>'
    f'<div class="value" style="color:#34d399">{energy["auroc"]:.4f}</div>'
    f'<div class="hint">unknown vs known</div></div>', unsafe_allow_html=True)
c2.markdown(
    f'<div class="stat-card"><div class="label">OOD AUROC - mahalanobis</div>'
    f'<div class="value" style="color:#c084fc">{maha["auroc"]:.4f}</div>'
    f'<div class="hint">embeddings, 32-d</div></div>', unsafe_allow_html=True)
c3.markdown(
    f'<div class="stat-card"><div class="label">FPR on benign @95% TPR</div>'
    f'<div class="value" style="color:#ef4444">{maha["benign_fpr"]*100:.2f}%</div>'
    f'<div class="hint">false alarms on normal traffic</div></div>', unsafe_allow_html=True)
c4.markdown(
    f'<div class="stat-card"><div class="label">Known-class accuracy</div>'
    f'<div class="value">{clf["acc"]*100:.1f}%</div>'
    f'<div class="hint">macro F1 {clf["f1"]:.3f}</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
f_hist = FIGURES / "ood_histogram.png"
f_roc = FIGURES / "auroc_curve.png"
if f_hist.exists():
    col1.image(str(f_hist), width="stretch")
if f_roc.exists():
    col2.image(str(f_roc), width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------- interactive OOD explorer
st.markdown('<div class="card"><span class="section-label">INTERACTIVE OOD '
            'SCORE EXPLORER (computed live from the run outputs)</span>',
            unsafe_allow_html=True)

scorer = st.selectbox("Scorer", ["energy", "mahalanobis"],
                      format_func=lambda s: "Energy OOD score" if s == "energy"
                      else "Mahalanobis distance")
tpr = st.slider("Desired recall of the unknown class (TPR)", 0.50, 0.99, 0.95, 0.01)

scores = samples[scorer]
unk_mask = samples["group"] == 2
known_mask = samples["group"] != 2
threshold = float(np.quantile(scores[unk_mask], tpr))
benign_fpr = float(np.mean(scores[samples["group"] == 0] >= threshold))
known_fpr = float(np.mean(scores[samples["group"] == 1] >= threshold))
sep = (float(scores[unk_mask].mean()) - float(scores[known_mask].mean())) / max(
    float(scores[known_mask].std()), 1e-9)

fig, ax = plt.subplots(figsize=(10, 4.4))
fig.patch.set_facecolor("#09090b")
ax.set_facecolor("#131316")
lo = min(scores.min(), threshold * 0.9)
hi = max(scores.max(), threshold * 1.1)
bins = np.linspace(lo, hi, 60)
for g, name in GROUP_NAMES.items():
    ax.hist(scores[samples["group"] == g], bins=bins, alpha=0.5,
            color=GROUP_COLORS[g], label=name)
ax.axvline(threshold, color="#fbbf24", ls="--", lw=1.6,
           label=f"threshold @{tpr:.0%} TPR = {threshold:.3f}")
ax.set_title(f"{scorer} score distribution  |  AUROC={metrics[scorer]['auroc']:.4f}  "
             f"separation={sep:.2f} std", color="#fff", fontsize=12, fontweight="bold")
ax.set_xlabel(f"{scorer} OOD score (higher = more OOD)", color="#e4e4e7")
ax.set_ylabel("count", color="#e4e4e7")
ax.tick_params(colors="#e4e4e7", labelsize=9)
for s in ax.spines.values():
    s.set_color("#27272a")
ax.grid(True, color="#1f1f23", linewidth=0.6)
ax.legend(facecolor="#131316", edgecolor="#27272a", labelcolor="#e4e4e7", fontsize=9)
st.pyplot(fig)

f1, f2, f3 = st.columns(3)
f1.markdown(f'<div class="stat-card"><div class="label">Threshold</div>'
            f'<div class="value">{threshold:.3f}</div><div class="hint">higher = OOD</div></div>',
            unsafe_allow_html=True)
f2.markdown(f'<div class="stat-card"><div class="label">FPR on benign</div>'
            f'<div class="value" style="color:#ef4444">{benign_fpr*100:.2f}%</div>'
            f'<div class="hint">false alarms on normal traffic</div></div>', unsafe_allow_html=True)
f3.markdown(f'<div class="stat-card"><div class="label">FPR on known attacks</div>'
            f'<div class="value" style="color:#fbbf24">{known_fpr*100:.2f}%</div>'
            f'<div class="hint">known attacks flagged as unknown</div></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ per-sample
st.markdown('<div class="card"><span class="section-label">PER-SAMPLE OOD SCORES '
            '(flagged = OOD at the selected threshold)</span>', unsafe_allow_html=True)
df = pd.DataFrame({
    "sample": samples["idx"],
    "group": [GROUP_NAMES[g] for g in samples["group"]],
    "class": samples["class_name"],
    "energy": samples["energy"],
    "mahalanobis": samples["mahalanobis"],
})
flag = df[scorer] >= threshold
df["flagged"] = np.where(flag, "OOD", "-")
table = df.assign(score=df[scorer]).sort_values("score", ascending=False)
show = pd.concat([
    table[table["group"] == "unknown (zero-day)"].head(25),
    table[table["group"] == "benign"].head(25),
    table[table["group"] == "known attack"].head(25),
], ignore_index=True)
st.dataframe(show[["sample", "group", "class", "energy", "mahalanobis",
                   "flagged"]], width="stretch", hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="card"><span class="section-label">KEY FINDINGS</span>'
    '<div class="muted" style="font-size:13px">'
    '&bull; A supervised IDS is overconfident on never-seen attacks: it still maps '
    'unknown samples into the known classes.<br>'
    '&bull; The OOD second stage separates the held-out zero-day (OOD AUROC '
    f'&asymp; {energy["auroc"]:.2f}-{maha["auroc"]:.2f}) while the classifier keeps '
    f'{clf["acc"]*100:.0f}% known-class accuracy.<br>'
    '&bull; At the 95%-recall operating point, normal-traffic false positives stay at '
    f'~{maha["benign_fpr"]*100:.1f}% (Mahalanobis).</div></div>',
    unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Zero-Day-Style OOD Detection &middot; manual energy + '
    'Mahalanobis scorers (pytorch-ood optional) &middot; held-out unknown: '
    'data-exfiltration</div>', unsafe_allow_html=True)
