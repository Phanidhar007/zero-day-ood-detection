"""Synthetic network-flow dataset with attack classes + a held-out "unknown".

No real data is required (and none is bundled). We synthesise a tabular
network-flow dataset with the following classes:

    benign, portscan, brute-force, malware-c2, data-exfiltration

The IDS model is deliberately trained WITHOUT seeing ``data-exfiltration``:
that class plays the role of a *zero-day style unknown attack* and only ever
appears at test time.

Each class gets a characteristic signature - a per-feature mean shift over a
baseline network profile - plus Gaussian noise. The known classes are
therefore learnable, while the held-out class lives in a different region of
feature (and hence embedding) space, which is exactly what an OOD detector
should flag.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "packets_per_sec",          # 0
    "bytes_per_flow",           # 1
    "avg_packet_size",          # 2
    "flow_duration_ms",         # 3
    "dst_ports_count",          # 4
    "syn_ratio",                # 5
    "failed_connections_ratio", # 6
    "payload_entropy",          # 7
    "connections_per_sec",      # 8
    "src_bytes_ratio",          # 9
    "tcp_flags_anomaly",        # 10
    "small_packets_ratio",      # 11
    "long_flow_ratio",          # 12
    "dns_requests_ratio",       # 13
    "auth_failures_per_sec",    # 14
    "data_volume_kb",           # 15
    "high_port_ratio",          # 16
    "beaconing_periodicity",    # 17
]

CLASS_NAMES = [
    "benign",
    "portscan",
    "brute-force",
    "malware-c2",
    "data-exfiltration",   # <-- held-out "zero-day" proxy, never in training
]
KNOWN_CLASS_NAMES = CLASS_NAMES[:4]
UNKNOWN_CLASS_NAME = CLASS_NAMES[-1]
N_CLASSES = len(CLASS_NAMES)
N_KNOWN = len(KNOWN_CLASS_NAMES)
N_FEATURES = len(FEATURE_NAMES)

# Per-feature noise scale (sigma). Synthetic features are unit-ish scale.
_SIGMA = np.array(
    [1.0, 1.5, 0.8, 2.0, 1.0, 0.6, 0.6, 0.8, 1.0, 0.8,
     0.6, 0.7, 0.7, 0.8, 0.8, 2.0, 0.7, 0.8],
    dtype=np.float64,
)

# Class signatures: per-feature mean shifts relative to a baseline profile,
# expressed in units of sigma. Designed so the known classes are separable and
# the held-out "data-exfiltration" class sits far from all known classes.
_SIGNATURES = {
    "benign": np.array(
        [0.2, 0.2, 0.3, 0.5, 0.3, -0.4, -0.5, 0.4, 0.2, 0.1,
         -0.3, 0.0, 0.2, 0.3, -0.4, 0.3, 0.1, -0.2]),
    "portscan": np.array(
        [-0.2, -0.6, -0.8, -1.2, 2.8, 3.0, -0.3, -1.0, 2.6, -0.5,
         0.8, 2.6, -1.4, 0.4, -0.2, -0.8, 0.3, 0.2]),
    "brute-force": np.array(
        [0.3, 0.2, 0.1, 0.6, 0.8, 0.6, 2.6, -0.6, 1.2, -0.2,
         0.4, 0.6, -0.2, 0.2, 3.0, 0.4, 0.2, 0.3]),
    "malware-c2": np.array(
        [-0.4, 1.4, 0.6, 1.8, 0.4, 0.2, 0.2, 0.9, 0.6, 0.3,
         0.6, -0.8, 1.4, -1.2, 0.5, 1.2, 1.8, 2.8]),
    # A "zero-day style" novel pattern: a broad, moderate elevation across
    # many flow dimensions (sustained high volume, many connections / ports,
    # high payload entropy, long flows). No single known class covers it, so
    # it is both ambiguous to the classifier and far from every known-class
    # embedding centroid.
    "data-exfiltration": np.array(
        [1.2, 2.4, 0.3, 1.8, 1.8, 1.2, 0.4, 1.8, 1.8, 1.8,
         0.8, 1.2, 1.6, 1.2, 0.2, 2.6, 0.8, 0.6]),
}


def make_traffic_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    benign_frac: float = 0.40,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n_samples`` labelled network-flow samples.

    Returns ``(X, y)`` where ``y`` indexes ``CLASS_NAMES``. Class 4
    (``data-exfiltration``) is the held-out unknown attack.
    """
    rng = np.random.RandomState(random_state)

    n_attack_total = int(np.round(n_samples * (1.0 - benign_frac)))
    n_attack_each = n_attack_total // (N_CLASSES - 1)
    counts = {
        "benign": n_samples - n_attack_each * (N_CLASSES - 1),
        "portscan": n_attack_each,
        "brute-force": n_attack_each,
        "malware-c2": n_attack_each,
        "data-exfiltration": n_attack_each,
    }

    X_list, y_list = [], []
    for class_id, name in enumerate(CLASS_NAMES):
        n = counts[name]
        mu = _SIGNATURES[name] * _SIGMA
        Xc = rng.normal(loc=mu, scale=_SIGMA, size=(n, N_FEATURES))
        X_list.append(Xc)
        y_list.append(np.full(n, class_id, dtype=np.int64))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    order = rng.permutation(len(X))
    return X[order], y[order]


def make_splits(
    n_samples: int = 5000,
    test_frac: float = 0.35,
    random_state: int = 42,
):
    """Split into a train set (KNOWN classes only) and a test set
    (KNOWN + the held-out UNKNOWN class).

    Returns ``(X_train, y_train, X_test, y_test, groups)`` where ``groups`` is
    an int array over the test set: 0 = benign, 1 = known attack,
    2 = unknown (held-out) attack.
    """
    X, y = make_traffic_dataset(n_samples=n_samples, random_state=random_state)
    rng = np.random.RandomState(random_state)

    known_mask = y < N_KNOWN
    unknown_mask = ~known_mask

    known_idx = np.where(known_mask)[0]
    rng.shuffle(known_idx)
    n_test_known = int(np.round(len(known_idx) * test_frac))
    test_known_idx = known_idx[:n_test_known]
    train_idx = known_idx[n_test_known:]

    test_idx = np.concatenate([test_known_idx, np.where(unknown_mask)[0]])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    groups = np.where(
        y_test == 0, 0, np.where(y_test < N_KNOWN, 1, 2),
    ).astype(np.int64)
    return X_train, y_train, X_test, y_test, groups
