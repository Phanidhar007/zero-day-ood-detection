"""Zero-Day-Style OOD Detection.

Train an IDS classifier on *known* attack classes, then build a separate OOD
scorer (energy-based and Mahalanobis distance on the penultimate-layer
embeddings) that flags inputs which don't resemble any known-class
distribution. A held-out attack class ("data-exfiltration") plays the role of
a zero-day / unknown attack at test time.

Package layout:
  data.py      - synthetic network-flow dataset + train/test splits with a
                 deliberately held-out "unknown" attack class
  model.py     - small torch MLP IDS classifier with an exposed embedding
                 (penultimate) layer
  ood.py       - manual energy-based + Mahalanobis OOD scorers
                 (pytorch-ood NOT required)
  evaluate.py  - OOD AUROC, FPR@threshold, known-vs-unknown histograms,
                 plots and metrics writer
"""

__version__ = "0.1.0"
