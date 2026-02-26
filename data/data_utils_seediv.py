"""
data_utils_seediv.py  (v2 — Improved)
======================================
Data loading and split utilities for SEED-IV EEG emotion classification.

Key additions over v1:
  • euclidean_alignment() — per-subject covariance whitening for the
    Subject-Independent (LOSO) paradigm; eliminates inter-subject
    amplitude distribution shifts that cause BN covariate-shift collapse.
  • load_seediv_SI_loso() now accepts si_euclidean_align flag.

Provides:
  • Subject-Dependent  (SD)  — train/test from the same subject
  • Subject-Independent (SI) — Leave-One-Subject-Out (LOSO)

All functions return (data, labels):
  data:   np.ndarray  (N, 62, 800)   float32
  labels: np.ndarray  (N,)            int64  → {0,1,2,3}
"""

import os
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────
N_SUBJECTS  = 15
N_SESSIONS  = 3
CLASS_NAMES = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}


# ─────────────────────────────────────────────────────────────────────────────
#  Euclidean Alignment  (He & Wu, 2019)
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    Per-subject Euclidean alignment to reference covariance.

    Aligns each subject's EEG data so that its mean covariance matrix
    equals the identity matrix.  This removes subject-level amplitude
    and covariance distribution shifts — the primary cause of poor
    generalisation in the LOSO paradigm — without requiring any
    labelled data from the test subject.

    Algorithm:
      1.  Compute C_i = (1/T) * x_i @ x_i^T  for each trial i
      2.  R̄ = mean(C_i)
      3.  R̄^{-1/2} via eigendecomposition
      4.  x̃_i = R̄^{-1/2} @ x_i  for all trials

    Parameters
    ----------
    X : np.ndarray  shape (N, 62, T)  — all trials for ONE subject

    Returns
    -------
    np.ndarray  shape (N, 62, T)  — aligned trials (float32)

    References
    ----------
    He & Wu (2019), "Transfer Learning for Brain-Computer Interfaces:
    A Euclidean Space Data Alignment Approach," IEEE T-BMEI.
    """
    N, C, T = X.shape

    # Step 1: compute per-trial covariance (unbiased)
    covs = np.einsum('nct,ndt->ncd', X, X) / T       # (N, C, C)

    # Step 2: mean covariance across trials
    R_mean = covs.mean(axis=0)                        # (C, C)

    # Step 3: symmetric matrix square-root inverse via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R_mean)
    # Guard against near-zero eigenvalues (numerical safety)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # (C, C)

    # Step 4: apply to all trials
    X_aligned = np.einsum('cd,ndt->nct', R_inv_sqrt, X)

    return X_aligned.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _npy_path(dataset_path: str, sub_id: int, session_id: int, kind: str) -> str:
    return os.path.join(dataset_path,
                        f'sub{sub_id}_session{session_id}_{kind}.npy')


def _load_one(dataset_path: str,
              sub_id: int,
              session_id: int) -> tuple:
    data_fp  = _npy_path(dataset_path, sub_id, session_id, 'data')
    label_fp = _npy_path(dataset_path, sub_id, session_id, 'label')
    if not os.path.exists(data_fp):
        raise FileNotFoundError(
            f'Missing: {data_fp}\nRun preprocess_data_seediv.py first.')
    data   = np.load(data_fp).astype(np.float32)
    labels = np.load(label_fp).astype(np.int64)
    return data, labels


def shuffle_data(data: np.ndarray,
                 label: np.ndarray) -> tuple:
    idx = np.random.permutation(len(data))
    return data[idx], label[idx]


def print_class_distribution(labels: np.ndarray, tag: str = '') -> None:
    total = labels.size
    parts = [f'{name}: {int((labels==k).sum())} ({100*(labels==k).mean():.1f}%)'
             for k, name in CLASS_NAMES.items()]
    prefix = f'[{tag}] ' if tag else ''
    print(f'{prefix}Classes → ' + ' | '.join(parts))


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Inverse-frequency class weights for weighted CrossEntropyLoss.
    Returns float32 array of shape (4,).
    """
    counts  = np.array([(labels == k).sum() for k in range(4)], dtype=np.float32)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 4.0    # normalise so mean = 1
    return weights.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Subject-Dependent (SD)
# ─────────────────────────────────────────────────────────────────────────────

def load_seediv_SD(dataset_path: str,
                   sub_id: int,
                   train_sessions: list = (1, 2),
                   test_sessions: list  = (3,),
                   shuffle: bool = True) -> tuple:
    """
    Subject-Dependent split.
    Default: sessions 1+2 → train, session 3 → test.
    Returns train_data, train_labels, test_data, test_labels.
    """
    def _collect(sessions):
        ds, ls = [], []
        for s in sessions:
            d, l = _load_one(dataset_path, sub_id, s)
            ds.append(d); ls.append(l)
        return np.concatenate(ds, 0), np.concatenate(ls, 0)

    tr_d, tr_l = _collect(train_sessions)
    te_d, te_l = _collect(test_sessions)

    if shuffle:
        tr_d, tr_l = shuffle_data(tr_d, tr_l)
        te_d, te_l = shuffle_data(te_d, te_l)

    print(f'[SD] Sub {sub_id:02d}  '
          f'train={list(train_sessions)}  test={list(test_sessions)}')
    print(f'  Train: {tr_d.shape}', end='  ')
    print_class_distribution(tr_l, 'train')
    print(f'  Test : {te_d.shape}', end='  ')
    print_class_distribution(te_l, 'test')

    return tr_d, tr_l, te_d, te_l


def load_seediv_SD_cv(dataset_path: str,
                      sub_id: int,
                      shuffle: bool = True) -> list:
    """
    3-fold cross-session CV.  Each fold: 1 session test, 2 train.
    Returns list of 3 tuples: (tr_d, tr_l, te_d, te_l).
    """
    folds = []
    for test_s in range(1, N_SESSIONS + 1):
        train_s = [s for s in range(1, N_SESSIONS + 1) if s != test_s]
        folds.append(load_seediv_SD(dataset_path, sub_id,
                                    train_sessions=train_s,
                                    test_sessions=[test_s],
                                    shuffle=shuffle))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
#  Subject-Independent (SI) — LOSO with optional Euclidean Alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_seediv_SI_loso(dataset_path: str,
                        test_subject: int,
                        sessions_to_use: list = (1, 2, 3),
                        shuffle: bool = True,
                        euclidean_align: bool = True) -> tuple:
    """
    Leave-One-Subject-Out (LOSO) split with optional Euclidean Alignment.

    Parameters
    ----------
    dataset_path     : str   Path to preprocessed .npy folder
    test_subject     : int   Held-out subject (1–15)
    sessions_to_use  : list  Sessions included (default: all 3)
    shuffle          : bool  Shuffle train and test independently
    euclidean_align  : bool  Apply per-subject EA before pooling.
                             Highly recommended for SI — corrects the
                             inter-subject covariance shift that causes
                             batch normalisation collapse and the
                             Fear over-prediction seen in v1 LOSO results.

    Returns
    -------
    train_data, train_labels, test_data, test_labels
    """
    train_data_list,  train_label_list  = [], []
    test_data_list,   test_label_list   = [], []

    for sub_id in range(1, N_SUBJECTS + 1):
        sub_data_parts, sub_label_parts = [], []
        for s in sessions_to_use:
            try:
                d, l = _load_one(dataset_path, sub_id, s)
                sub_data_parts.append(d)
                sub_label_parts.append(l)
            except FileNotFoundError as e:
                print(f'  [WARN] {e}')

        if not sub_data_parts:
            continue

        sub_data  = np.concatenate(sub_data_parts,  axis=0)   # (N_s, 62, 800)
        sub_label = np.concatenate(sub_label_parts, axis=0)   # (N_s,)

        # Apply Euclidean Alignment per subject (independently)
        if euclidean_align:
            sub_data = euclidean_alignment(sub_data)

        if sub_id == test_subject:
            test_data_list.append(sub_data)
            test_label_list.append(sub_label)
        else:
            train_data_list.append(sub_data)
            train_label_list.append(sub_label)

    if not train_data_list or not test_data_list:
        raise RuntimeError(
            f'No data for LOSO fold (test_subject={test_subject}).')

    train_data   = np.concatenate(train_data_list,  axis=0)
    train_labels = np.concatenate(train_label_list, axis=0)
    test_data    = np.concatenate(test_data_list,   axis=0)
    test_labels  = np.concatenate(test_label_list,  axis=0)

    if shuffle:
        train_data, train_labels = shuffle_data(train_data,  train_labels)
        test_data,  test_labels  = shuffle_data(test_data,   test_labels)

    ea_tag = 'EA=ON' if euclidean_align else 'EA=OFF'
    print(f'[SI-LOSO] Test sub={test_subject:02d}  '
          f'sessions={list(sessions_to_use)}  {ea_tag}')
    print(f'  Train: {train_data.shape}', end='  ')
    print_class_distribution(train_labels, 'train')
    print(f'  Test : {test_data.shape}',  end='  ')
    print_class_distribution(test_labels, 'test')

    return train_data, train_labels, test_data, test_labels


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(dataset_path: str) -> None:
    print('\n' + '=' * 65)
    print('SEED-IV Dataset Inspection')
    print('=' * 65)
    grand_total = 0
    for sub_id in range(1, N_SUBJECTS + 1):
        for s in range(1, N_SESSIONS + 1):
            dp = _npy_path(dataset_path, sub_id, s, 'data')
            lp = _npy_path(dataset_path, sub_id, s, 'label')
            if os.path.exists(dp) and os.path.exists(lp):
                d = np.load(dp); l = np.load(lp)
                grand_total += d.shape[0]
                dist = {k: int((l == k).sum()) for k in range(4)}
                print(f'  sub{sub_id:02d}_s{s}  {d.shape}  '
                      f'N:{dist[0]} S:{dist[1]} F:{dist[2]} H:{dist[3]}')
            else:
                print(f'  sub{sub_id:02d}_s{s}  [MISSING]')
    print(f'\nTotal windows: {grand_total}')
    print('=' * 65 + '\n')


if __name__ == '__main__':
    inspect_dataset('dataset/seediv')
