"""
preprocess_data_seediv.py  (v2 — Improved)
==========================================
Converts SEED-IV raw EEG (.mat) files into fixed-length .npy arrays
ready for TransNet training.

Key improvements over v1:
  • 50% OVERLAPPING windows (stride = WIN_SAMPLES // 2 = 400 samples)
    → ~14 windows per 30-second trial instead of 7
    → doubles the training set size for the same data
  • Per-CHANNEL z-score normalization (not global across all channels)
    → preserves inter-channel amplitude relationships that the spatial
       convolution layer needs to learn
  • h5py fallback for MATLAB v7.3 format files (unchanged)
  • Sorts .mat files by subject index (unchanged)

Pipeline
--------
1. Load cz_eeg1 … cz_eeg24 from each session .mat file   (62 × T_i)
2. Butterworth bandpass filter  1–75 Hz  (200 Hz Fs, order 4)
3. Per-CHANNEL z-score normalisation  (per trial)
4. Overlapping 4-second windows  (800 samples, stride 400)
5. Save  sub{i}_session{j}_data.npy  and  sub{i}_session{j}_label.npy

Outputs
-------
  <SAVE_PATH>/
    sub1_session1_data.npy   shape: (N_windows, 62, 800)
    sub1_session1_label.npy  shape: (N_windows,)
    ...

Usage
-----
  python preprocess_data_seediv.py

Edit DATA_ROOT and SAVE_PATH below.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import scipy.signal as signal

# ─────────────────────────────────────────────────────────────────────────────
#  USER CONFIGURATION  ← edit these two lines
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = 'data/SEED-IV/eeg_raw_data'   # folder containing sub-dirs 1/, 2/, 3/
SAVE_PATH = 'dataset/seediv'               # output .npy directory
# ─────────────────────────────────────────────────────────────────────────────

# ── Signal constants ─────────────────────────────────────────────────────────
SFREQ        = 200          # distributed sampling rate (Hz)
WIN_SAMPLES  = 800          # 4 s × 200 Hz
WIN_STRIDE   = 400          # 50% overlap → 2× more windows than v1
N_CHANNELS   = 62
N_SESSIONS   = 3
N_TRIALS     = 24

# ── Bandpass filter ─────────────────────────────────────────────────────────
BANDPASS_LOW  = 1.0         # Hz
BANDPASS_HIGH = 75.0        # Hz
FILTER_ORDER  = 4

# ── Ground-truth labels (from ReadMe.txt) ────────────────────────────────────
#    0=Neutral, 1=Sad, 2=Fear, 3=Happy
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}


# ─────────────────────────────────────────────────────────────────────────────
#  DSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_bandpass_filter(lowcut, highcut, fs, order=4):
    nyq  = fs / 2.0
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass')
    return b, a


def apply_bandpass(eeg: np.ndarray, b, a) -> np.ndarray:
    """Zero-phase bandpass.  eeg: (C, T)"""
    return signal.filtfilt(b, a, eeg, axis=1)


def per_channel_zscore(eeg: np.ndarray) -> np.ndarray:
    """
    Per-CHANNEL z-score along the time axis.  eeg: (C, T)

    IMPORTANT: v1 used global z-score (single mean/std across all C×T
    values), which destroyed the amplitude differences between channels
    that the spatial convolution layer needs.  This per-channel version
    normalises each channel independently, preserving inter-channel
    structure.
    """
    mean = eeg.mean(axis=1, keepdims=True)        # (C, 1)
    std  = eeg.std(axis=1,  keepdims=True)        # (C, 1)
    std  = np.where(std < 1e-8, 1e-8, std)
    return (eeg - mean) / std


def segment_trial_overlapping(eeg: np.ndarray,
                               win_samples: int,
                               stride: int) -> np.ndarray:
    """
    Slice one trial into OVERLAPPING windows.
    eeg:   (C, T)
    Returns array of shape (n_windows, C, win_samples).

    50% overlap (stride = win_samples // 2) roughly doubles the number
    of training windows compared to non-overlapping segmentation.
    """
    C, T = eeg.shape
    starts = range(0, T - win_samples + 1, stride)
    segs   = [eeg[:, s:s + win_samples] for s in starts]
    if len(segs) == 0:
        return np.empty((0, C, win_samples), dtype=np.float32)
    return np.stack(segs, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  .mat loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mat_session(mat_path: str) -> dict:
    """
    Load a .mat file → {trial_idx (0-based): eeg_array (62 × T)}.
    Handles both old scipy and MATLAB v7.3 (h5py) formats.
    """
    try:
        mat = sio.loadmat(mat_path)
    except NotImplementedError:
        import h5py
        mat = {}
        with h5py.File(mat_path, 'r') as f:
            for k in f.keys():
                if k.startswith('cz_eeg'):
                    mat[k] = np.array(f[k]).T
    trials = {}
    for i in range(1, N_TRIALS + 1):
        key = f'cz_eeg{i}'
        if key in mat:
            trials[i - 1] = np.array(mat[key], dtype=np.float64)
    return trials


# ─────────────────────────────────────────────────────────────────────────────
#  Per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_subject_session(mat_path: str, session_id: int,
                             b, a) -> tuple:
    """
    Full pipeline for one (subject × session) .mat file.
    Returns (data, labels):
      data:   (N_windows, 62, 800)
      labels: (N_windows,)
    """
    labels_for_session = SESSION_LABELS[session_id]
    trials = load_mat_session(mat_path)

    all_segments, all_labels = [], []

    for trial_idx in range(N_TRIALS):
        if trial_idx not in trials:
            print(f'  [WARN] Trial {trial_idx+1} missing in {mat_path}')
            continue

        eeg = trials[trial_idx]                       # (62, T)

        # 1. Bandpass filter
        eeg = apply_bandpass(eeg, b, a)

        # 2. Per-channel z-score
        eeg = per_channel_zscore(eeg)

        # 3. Overlapping segmentation (50% overlap)
        segs = segment_trial_overlapping(eeg, WIN_SAMPLES, WIN_STRIDE)

        if segs.shape[0] == 0:
            print(f'  [WARN] Trial {trial_idx+1} too short '
                  f'(T={eeg.shape[1]} < {WIN_SAMPLES})')
            continue

        emotion = labels_for_session[trial_idx]
        all_segments.append(segs)
        all_labels.append(np.full(segs.shape[0], emotion, dtype=np.int64))

    if len(all_segments) == 0:
        return (np.empty((0, N_CHANNELS, WIN_SAMPLES), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    return np.concatenate(all_segments, 0), np.concatenate(all_labels, 0)


def get_sorted_mat_files(session_dir: str) -> list:
    """Return .mat paths sorted by numeric subject index prefix."""
    return sorted(
        glob.glob(os.path.join(session_dir, '*.mat')),
        key=lambda p: int(os.path.basename(p).split('_')[0])
    )


def print_class_dist(labels: np.ndarray, prefix=''):
    names  = {0:'Neutral', 1:'Sad', 2:'Fear', 3:'Happy'}
    total  = labels.size
    parts  = [f'{v}:{int((labels==k).sum())}({100*(labels==k).mean():.0f}%)'
              for k, v in names.items()]
    print(f'{prefix}  ' + ' | '.join(parts))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)
    b, a = build_bandpass_filter(BANDPASS_LOW, BANDPASS_HIGH, SFREQ, FILTER_ORDER)

    print('=' * 65)
    print('SEED-IV Preprocessing  (v2 — Overlapping Windows)')
    print(f'  Fs           : {SFREQ} Hz')
    print(f'  Window       : {WIN_SAMPLES} samples  ({WIN_SAMPLES/SFREQ:.1f} s)')
    print(f'  Stride       : {WIN_STRIDE} samples  ({WIN_STRIDE/SFREQ:.1f} s)  '
          f'→ 50% overlap')
    print(f'  Bandpass     : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz')
    print(f'  Normalisation: per-channel z-score')
    print(f'  Source       : {DATA_ROOT}')
    print(f'  Output       : {SAVE_PATH}')
    print('=' * 65)

    total_windows = 0

    for session_id in range(1, N_SESSIONS + 1):
        session_dir = os.path.join(DATA_ROOT, str(session_id))
        mat_files   = get_sorted_mat_files(session_dir)
        print(f'\n── Session {session_id}  ({len(mat_files)} files) ──')

        if not mat_files:
            print(f'  [ERROR] No .mat files in {session_dir}')
            continue

        for sub_idx, mat_path in enumerate(mat_files, start=1):
            fname = os.path.basename(mat_path)
            print(f'\n  Sub {sub_idx:02d} | {fname}')

            data, labels = process_subject_session(mat_path, session_id, b, a)

            if data.shape[0] == 0:
                print('  [ERROR] No segments extracted — skipping.')
                continue

            tag      = f'sub{sub_idx}_session{session_id}'
            data_fp  = os.path.join(SAVE_PATH, f'{tag}_data.npy')
            label_fp = os.path.join(SAVE_PATH, f'{tag}_label.npy')
            np.save(data_fp,  data)
            np.save(label_fp, labels)

            total_windows += data.shape[0]
            print(f'  data  → {data_fp}   shape: {data.shape}')
            print(f'  label → {label_fp}  shape: {labels.shape}')
            print_class_dist(labels, prefix=f'  Sub{sub_idx:02d}')

    print('\n' + '=' * 65)
    print(f'Done.  Total windows: {total_windows}')
    windows_per_trial_approx = (6000 - WIN_SAMPLES) // WIN_STRIDE + 1
    print(f'Expected ~{windows_per_trial_approx} windows/trial  '
          f'(vs 7 in v1 without overlap)')
    print('=' * 65)


if __name__ == '__main__':
    main()
