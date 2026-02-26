"""
train_transnet_seediv_SI.py  (v2 — Improved)
==============================================
Subject-Independent (LOSO) emotion classification using TransNet on SEED-IV.

Key improvements over v1:
  1. Euclidean Alignment (EA) per subject before training.
     Corrects inter-subject covariance distribution shifts — the primary
     reason LOSO accuracy was only 39.42% in v1.
     Each subject's EEG trials are independently whitened so the mean
     covariance matrix equals identity.  This is applied uniformly to
     both train and test subjects so no test labels are required.
  2. Weighted CrossEntropyLoss — same as SD script.
  3. All v1 improvements retained: AMP, gradient clipping, cosine LR.

Protocol: Leave-One-Subject-Out (LOSO)
  Train: all 3 sessions × 14 subjects (EA applied per subject)
  Test:  all 3 sessions × 1 held-out subject (EA applied)

Usage
-----
  python train_transnet_seediv_SI.py
  python train_transnet_seediv_SI.py --config seediv_transnet.yaml
  python train_transnet_seediv_SI.py --test_subjects 1 5 10
  python train_transnet_seediv_SI.py --no_ea          # disable alignment
"""

import os, sys, time, random, argparse, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TransNet          import TransNet
from baseModel         import baseModel
from dataset_seediv    import SeedIVDataset
from data_utils_seediv import (load_seediv_SI_loso, inspect_dataset)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def ts():
    return time.strftime('%Y-%m-%d--%H-%M', time.localtime())


def build_net(config):
    net = TransNet(**config['network_args'])
    n   = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'  TransNet — trainable params: {n:,}')
    return net


def save_yaml(path, cfg):
    with open(path, 'w') as f:
        yaml.dump(cfg, f)


# ─────────────────────────────────────────────────────────────────────────────
#  Single LOSO fold
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_fold(tr_d, tr_l, te_d, te_l, config, out_path, test_sub):
    os.makedirs(out_path, exist_ok=True)
    save_yaml(os.path.join(out_path, 'config.yaml'), config)

    train_ds = SeedIVDataset(tr_d, tr_l)
    test_ds  = SeedIVDataset(te_d, te_l)
    train_ds.summary(tag=f'TRAIN (test={test_sub:02d})')
    test_ds.summary( tag=f'TEST  (test={test_sub:02d})')

    set_seed(config['random_seed'])
    net = build_net(config)

    # ── Weighted CE loss ─────────────────────────────────────────────────
    use_weighted = config.get('use_weighted_loss', True)
    if use_weighted:
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights   = train_ds.class_weights().to(device)
        loss_func = nn.CrossEntropyLoss(weight=weights)
        print(f'  Weighted CE — weights: {weights.cpu().numpy().round(3)}')
    else:
        loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    model = baseModel(net, config, optimizer, loss_func,
                      result_savepath=out_path)
    model.train_test(train_ds, test_ds)

    # ── Final evaluation ─────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(os.path.join(out_path, 'model.pth'),
                                   map_location='cpu'))
    net = net.to(device).eval()
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=config['batch_size'],
                        shuffle=False, num_workers=0)
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.float().to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = net(xb)
            preds.extend(out.argmax(1).cpu().tolist())
            actuals.extend(yb.tolist())

    acc   = accuracy_score(actuals, preds)
    kappa = cohen_kappa_score(actuals, preds)
    cm    = confusion_matrix(actuals, preds, labels=[0, 1, 2, 3])
    return dict(acc=acc, kappa=kappa, cm=cm)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config, test_subjects=None, euclidean_align=None):
    data_path   = config['data_path']
    out_folder  = config['out_folder']
    si_sessions = config.get('si_sessions', [1, 2, 3])
    run_ts      = ts()

    # EA flag: CLI > config > default True
    if euclidean_align is None:
        euclidean_align = config.get('si_euclidean_align', True)

    if test_subjects is None:
        test_subjects = list(range(1, 16))

    inspect_dataset(data_path)

    all_results = {}
    si_out_root = os.path.join(out_folder, 'TransNet', 'SI', run_ts)

    for test_sub in test_subjects:
        print('\n' + '═' * 65)
        print(f'  LOSO fold: test_subject={test_sub:02d}  EA={euclidean_align}')
        print('═' * 65)

        tr_d, tr_l, te_d, te_l = load_seediv_SI_loso(
            data_path,
            test_subject=test_sub,
            sessions_to_use=si_sessions,
            euclidean_align=euclidean_align)

        fold_out = os.path.join(si_out_root, f'subject_{test_sub:02d}')
        res      = run_loso_fold(tr_d, tr_l, te_d, te_l,
                                 config, fold_out, test_sub)

        print(f'\n  Result (test_sub={test_sub:02d}): '
              f'Acc={res["acc"]:.4f}  κ={res["kappa"]:.4f}')
        print('  Confusion matrix (rows=true, cols=pred):')
        print('  Labels: N=Neutral  S=Sad  F=Fear  H=Happy')
        print(res['cm'])
        all_results[test_sub] = res

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '═' * 65)
    print('SUBJECT-INDEPENDENT (LOSO) SUMMARY')
    print('═' * 65)
    accs   = [r['acc']   for r in all_results.values()]
    kappas = [r['kappa'] for r in all_results.values()]
    for sub_id, res in all_results.items():
        print(f'  Sub {sub_id:02d}: Acc={res["acc"]*100:.2f}%  '
              f'κ={res["kappa"]:.4f}')
    if accs:
        print(f'\n  Mean Acc : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%')
        print(f'  Mean κ   : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}')

    summary_path = os.path.join(si_out_root, f'SI_summary_{run_ts}.txt')
    os.makedirs(si_out_root, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write('EEG-TransNet SEED-IV Subject-Independent (LOSO)\n')
        f.write(f'Timestamp        : {run_ts}\n')
        f.write(f'Sessions         : {si_sessions}\n')
        f.write(f'Euclidean Align  : {euclidean_align}\n\n')
        for sub_id, res in all_results.items():
            f.write(f'Sub {sub_id:02d}: acc={res["acc"]:.6f}  '
                    f'kappa={res["kappa"]:.6f}\n')
        if accs:
            f.write(f'\nMean acc  : {np.mean(accs):.6f}  '
                    f'Std: {np.std(accs):.6f}\n')
            f.write(f'Mean kappa: {np.mean(kappas):.6f}  '
                    f'Std: {np.std(kappas):.6f}\n')
    print(f'\n  Summary → {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='seediv_transnet.yaml')
    parser.add_argument('--test_subjects', type=int, nargs='+', default=None)
    parser.add_argument('--no_ea', action='store_true',
                        help='Disable Euclidean alignment (for ablation)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.full_load(f)

    main(config,
         test_subjects=args.test_subjects,
         euclidean_align=(not args.no_ea))
