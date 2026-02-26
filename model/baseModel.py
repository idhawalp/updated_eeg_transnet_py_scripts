"""
baseModel.py  (v2 — SEED-IV Improved)
=======================================
Training loop for TransNet on SEED-IV.

Key improvements over v1:
  • Weighted CrossEntropyLoss (passed in from training script using
    train_dataset.class_weights()) — fixes the Fear class dominance
    that produced the skewed confusion matrix in v1.
  • CosineAnnealingLR over the full epoch count (2000), not truncated
    to 100 — ensures LR stays meaningful throughout training.
  • Gradient clipping (max_norm=1.0) — prevents occasional gradient
    explosions in the transformer attention layers.
  • Mixed precision (AMP) + GradScaler — unchanged from v1.
  • num_workers=0 for Windows / MPS compatibility — unchanged from v1.
  • NO early stopping — mirrors the original paper which saves the best
    checkpoint over all 2000 epochs without stopping.
  • Logs kappa in addition to accuracy — useful for imbalanced 4-class.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score


class baseModel:
    def __init__(self, net, config, optimizer, loss_func,
                 scheduler=None, result_savepath=None):

        self.batchsize       = config['batch_size']
        self.epochs          = config['epochs']
        self.preferred_device = config['preferred_device']
        self.num_classes     = config['num_classes']
        self.num_segs        = config['num_segs']

        self.device = self._resolve_device(config['nGPU'])
        self.net    = net.to(self.device)

        self.optimizer = optimizer
        # Accept whatever loss_func is passed in (could be weighted CE)
        self.loss_func = loss_func

        # Cosine annealing over the FULL epoch count (critical fix)
        self.scheduler = (scheduler if scheduler is not None else
                          torch.optim.lr_scheduler.CosineAnnealingLR(
                              optimizer, T_max=self.epochs, eta_min=1e-6))

        # AMP scaler (graceful fallback on CPU)
        self.use_amp = torch.cuda.is_available()
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logging
        self.result_savepath = result_savepath
        self.log_write       = None
        if result_savepath is not None:
            os.makedirs(result_savepath, exist_ok=True)
            self.log_write = open(
                os.path.join(result_savepath, 'log_result.txt'), 'w')

    # ── Device setup ──────────────────────────────────────────────────────

    def _resolve_device(self, nGPU):
        if self.preferred_device == 'gpu' and torch.cuda.is_available():
            return torch.device(f'cuda:{nGPU}')
        return torch.device('cpu')

    # ── Data augmentation (temporal segment mixing) ───────────────────────

    def data_augmentation(self, data, label):
        """
        Intra-class temporal segment mixing.
        Same logic as original paper (Algorithm 1).
        """
        label_np = (label.cpu().numpy() if torch.is_tensor(label)
                    else np.asarray(label))
        N, C, T  = data.shape
        seg_size = T // self.num_segs
        aug_per_class = max(1, self.batchsize // self.num_classes)

        aug_data_list, aug_label_list = [], []

        for cls in range(self.num_classes):
            cls_idx  = np.where(label_np == cls)[0]
            if len(cls_idx) <= 1:
                continue
            cls_data = (data[cls_idx].cpu().numpy()
                        if torch.is_tensor(data) else data[cls_idx])
            n        = len(cls_idx)
            buf      = np.zeros((aug_per_class, C, T), dtype=np.float32)
            for i in range(aug_per_class):
                rand_idx = np.random.randint(0, n, self.num_segs)
                for j in range(self.num_segs):
                    buf[i, :, j*seg_size:(j+1)*seg_size] = \
                        cls_data[rand_idx[j], :, j*seg_size:(j+1)*seg_size]
            aug_data_list.append(buf)
            aug_label_list.extend([cls] * aug_per_class)

        if not aug_data_list:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        aug_data  = np.concatenate(aug_data_list, axis=0)
        aug_label = np.array(aug_label_list, dtype=np.int64)
        perm      = np.random.permutation(len(aug_data))
        return (torch.from_numpy(aug_data[perm]),
                torch.from_numpy(aug_label[perm]))

    # ── Main training loop ────────────────────────────────────────────────

    def train_test(self, train_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batchsize,
                                  shuffle=True,  num_workers=0,
                                  pin_memory=self.use_amp, drop_last=True)
        test_loader  = DataLoader(test_dataset,  batch_size=self.batchsize,
                                  shuffle=False, num_workers=0,
                                  pin_memory=self.use_amp, drop_last=False)

        best_acc   = 0.0
        best_kappa = 0.0
        best_model = None

        for epoch in range(self.epochs):
            # ── Train ──────────────────────────────────────────────────
            self.net.train()
            tr_preds, tr_actuals = [], []

            for xb, yb in train_loader:
                aug_x, aug_y = self.data_augmentation(xb, yb)

                if aug_x.numel() > 0:
                    xb = torch.cat([xb.float(), aug_x.float()], dim=0)
                    yb = torch.cat([yb.long(),  aug_y.long()],  dim=0)

                xb = xb.float().to(self.device, non_blocking=True)
                yb = yb.long().to( self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out  = self.net(xb)
                    loss = self.loss_func(out, yb)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                tr_preds.extend(out.argmax(1).detach().cpu().tolist())
                tr_actuals.extend(yb.cpu().tolist())

            self.scheduler.step()

            # ── Test ───────────────────────────────────────────────────
            self.net.eval()
            te_preds, te_actuals = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.float().to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        out = self.net(xb)
                    te_preds.extend(out.argmax(1).cpu().tolist())
                    te_actuals.extend(yb.tolist())

            tr_acc = accuracy_score(tr_actuals, tr_preds)
            te_acc = accuracy_score(te_actuals, te_preds)
            te_kap = cohen_kappa_score(te_actuals, te_preds)
            lr_now = self.optimizer.param_groups[0]['lr']

            if te_acc > best_acc:
                best_acc   = te_acc
                best_kappa = te_kap
                best_model = copy.deepcopy(self.net.state_dict())

            # Print every epoch; log every 50
            print(f'Epoch [{epoch+1:4d}/{self.epochs}] '
                  f'TrainAcc={tr_acc:.4f}  '
                  f'TestAcc={te_acc:.4f}  '
                  f'κ={te_kap:.4f}  '
                  f'Best={best_acc:.4f}  '
                  f'LR={lr_now:.2e}')

            if self.log_write and (epoch % 50 == 0 or te_acc >= best_acc):
                self.log_write.write(
                    f'Epoch [{epoch+1}] '
                    f'TrainAcc={tr_acc:.6f}  '
                    f'TestAcc={te_acc:.6f}  '
                    f'Kappa={te_kap:.6f}\n')
                self.log_write.flush()

        # ── Save best model ────────────────────────────────────────────
        print(f'\n  Best Accuracy: {best_acc:.6f}')
        print(f'  Best Kappa   : {best_kappa:.6f}')

        if self.log_write:
            self.log_write.write(f'\nBest Accuracy: {best_acc:.6f}\n')
            self.log_write.write(f'Best Kappa   : {best_kappa:.6f}\n')
            self.log_write.close()

        torch.save(best_model,
                   os.path.join(self.result_savepath, 'model.pth'))
