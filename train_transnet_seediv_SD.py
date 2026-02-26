"""
train_transnet_seediv_SD.py  (v2 — Improved)
==============================================
Subject-Dependent emotion classification using TransNet on SEED-IV.

Key improvement over v1:
  • Weighted CrossEntropyLoss — computed from the training set class
    distribution and passed to baseModel.  Prevents the model from
    defaulting to Fear (the most frequent test-session class).

Protocol:
  Train sessions : 1 + 2  (pooled)
  Test session   : 3
  Subjects       : 1–15  (or subset via --subjects)

Usage
-----
  python train_transnet_seediv_SD.py
  python train_transnet_seediv_SD.py --config seediv_transnet.yaml
  python train_transnet_seediv_SD.py --subjects 1 4 7
  python train_transnet_seediv_SD.py --crossval      # 3-fold cross-session
"""

import os, sys, time, random, argparse, yaml, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TransNet          import TransNet
from baseModel         import baseModel
from dataset_seediv    import SeedIVDataset
from data_utils_seediv import (load_seediv_SD, load_seediv_SD_cv,
                                inspect_dataset)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False   # reproducibility > speed


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
#  Single fold
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(tr_d, tr_l, te_d, te_l, config, out_path, tag=''):
    os.makedirs(out_path, exist_ok=True)
    save_yaml(os.path.join(out_path, 'config.yaml'), config)

    train_ds = SeedIVDataset(tr_d, tr_l)
    test_ds  = SeedIVDataset(te_d, te_l)
    train_ds.summary(tag=f'TRAIN {tag}')
    test_ds.summary( tag=f'TEST  {tag}')

    set_seed(config['random_seed'])
    net = build_net(config)

    # ── Weighted CE loss (key improvement) ──────────────────────────────
    use_weighted = config.get('use_weighted_loss', True)
    if use_weighted:
        weights   = train_ds.class_weights().to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        loss_func = nn.CrossEntropyLoss(weight=weights)
        print(f'  Weighted CE — class weights: {weights.cpu().numpy().round(3)}')
    else:
        loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    model = baseModel(net, config, optimizer, loss_func,
                      result_savepath=out_path)
    model.train_test(train_ds, test_ds)

    # ── Final evaluation on best saved model ────────────────────────────
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

def main(config, subjects=None, cross_val=False):
    data_path      = config['data_path']
    out_folder     = config['out_folder']
    train_sessions = config.get('sd_train_sessions', [1, 2])
    test_sessions  = config.get('sd_test_sessions',  [3])
    run_ts         = ts()

    if subjects is None:
        subjects = list(range(1, 16))

    inspect_dataset(data_path)

    all_results = {}

    for sub_id in subjects:
        print('\n' + '═' * 65)
        print(f'  Subject {sub_id:02d} / 15   [Subject-Dependent]')
        print('═' * 65)
        sub_root = os.path.join(out_folder, 'TransNet', f'sub{sub_id:02d}', run_ts)

        if cross_val:
            print('  Mode: 3-fold cross-session CV')
            folds    = load_seediv_SD_cv(data_path, sub_id)
            fold_acc = []
            for fi, (tr_d, tr_l, te_d, te_l) in enumerate(folds):
                res = run_fold(tr_d, tr_l, te_d, te_l, config,
                               os.path.join(sub_root, f'fold{fi+1}'),
                               tag=f'S{sub_id} F{fi+1}')
                fold_acc.append(res['acc'])
                print(f'  Fold {fi+1}: Acc={res["acc"]:.4f}  κ={res["kappa"]:.4f}')
            mean_acc = float(np.mean(fold_acc))
            print(f'\n  CV mean acc (sub{sub_id:02d}): {mean_acc:.4f}')
            all_results[sub_id] = dict(acc=mean_acc, fold_accs=fold_acc)

        else:
            tr_d, tr_l, te_d, te_l = load_seediv_SD(
                data_path, sub_id,
                train_sessions=train_sessions,
                test_sessions=test_sessions)
            res = run_fold(tr_d, tr_l, te_d, te_l, config,
                           sub_root, tag=f'S{sub_id}')
            print(f'\n  Result (sub{sub_id:02d}): '
                  f'Acc={res["acc"]:.4f}  κ={res["kappa"]:.4f}')
            print('  Confusion matrix (N S F H):')
            print(res['cm'])
            all_results[sub_id] = res

    # ── Summary ──────────────────────────────────────────────────────────
    print('\n' + '═' * 65)
    print('SUBJECT-DEPENDENT SUMMARY')
    print('═' * 65)
    accs = [r['acc'] for r in all_results.values()]
    for sub_id, res in all_results.items():
        print(f'  Sub {sub_id:02d}: {res["acc"]*100:.2f}%')
    if accs:
        print(f'\n  Mean ± Std : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%')

    summary_path = os.path.join(out_folder, 'TransNet',
                                f'SD_summary_{run_ts}.txt')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write('EEG-TransNet SEED-IV Subject-Dependent\n')
        f.write(f'Timestamp: {run_ts}\n')
        f.write(f'Train: {train_sessions}  Test: {test_sessions}\n\n')
        for sub_id, res in all_results.items():
            f.write(f'Sub {sub_id:02d}: acc={res["acc"]:.6f}\n')
        if accs:
            f.write(f'\nMean={np.mean(accs):.6f}  Std={np.std(accs):.6f}\n')
    print(f'\n  Summary → {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='seediv_transnet.yaml')
    parser.add_argument('--subjects', type=int, nargs='+', default=None)
    parser.add_argument('--crossval', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.full_load(f)

    main(config,
         subjects=args.subjects,
         cross_val=args.crossval or config.get('sd_cross_val', False))
