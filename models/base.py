"""
Note: torch.autograd.set_detect_anomaly(True) makes the calculation very slow
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from benchmark.const import CONFIG_PATH_SMILES
from benchmark.utils import get_config, calc_rmse
from utils import inf_iterator

smiles_encoding = get_config(CONFIG_PATH_SMILES)
_MODEL_DICT = {}
get_model = lambda cfg: _MODEL_DICT[cfg]


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls

    return decorator


@register_model('GNN')
class GNN:
    def __init__(self, config=None):
        self.config = config
        self.val_losses, self.val_roc, self.mean, self.mad = [], [], 0.0, 1.0
        self.model, self.critic, self.loss_fn, self.teacher_model = None, None, None, None
        self.optimizer, self.lr_scheduler, self.device, self.best_path = None, None, None, None

    def train(self, x_train: List[Data], y_train: List[float], x_val, y_val, epochs=100, batch_size=64, base_path=None, classification=False):
        t = tqdm(range(epochs))
        y_val = torch.tensor(y_val)
        train_loader = graphs_to_loader(x_train, y_train, batch_size=batch_size + 1 if len(x_train) % batch_size == 1 else batch_size)
        val_loader = DataLoader(x_val, batch_size=batch_size + 1 if len(x_val) % batch_size == 1 else batch_size, shuffle=False)

        for e in t:
            loss = self._one_epoch(train_loader) / len(x_train)
            val_pred = self.predict(val_loader)
            if classification:
                val_score = self.classification_test(val_pred, y_val.tolist(), self.model.num_class)
                self.val_roc.append(val_score)
            else:
                val_score = self.loss_fn(squeeze_if_needed(val_pred), y_val)
                self.val_losses.append(val_score)
            t.set_description(f"Epoch {e + 1} | Train {loss:.3f} | Val {val_score:.3f} |  Best_Val {max(self.val_roc) if classification else min(self.val_losses):.3f} "
                              f"|  Lr(1e-4): {self.optimizer.param_groups[0]['lr'] * 1e4:.3f}")

            if self.lr_scheduler is not None: self.lr_scheduler.step()
            if (classification and val_score >= max(self.val_roc)) or (not classification and val_score <= min(self.val_losses)):
                torch.save([self.model.state_dict(), val_score], base_path)

    def _one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            batch.to(self.device)
            y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
            target = (batch.y - self.mean) / self.mad
            if self.model.num_class > 1:
                target = target.reshape(-1, self.model.num_class)
            loss = self.loss_fn(squeeze_if_needed(y_hat), squeeze_if_needed(target))
            if loss.dim() > 1:
                valids = (target != 0.5)
                loss = (loss * valids).sum() / valids.sum()
            if not loss > 0:
                print(idx, y_hat[0], batch.y[0], target[0], loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * len(batch.y)
        return total_loss

    def InstructLearning(self, x_train, y_train, x_val, y_val, x_unlabeled, x_test, y_test, max_iters, val_freq, critic_iters, patience, batch_size, instruct_lr, min_lr,
                         max_grad_norm, update_freq, only_positive, critic_loss_weight, unlabeled_weight, soft_label, labeled_weight, y_unlabeled=None, classification=False,
                         loss_path=None, save_prob=False):
        for g in self.optimizer.param_groups:  # reset learning rate
            g['lr'] = instruct_lr

        test_score = -1.0
        t = tqdm(range(max_iters))
        y_val = torch.tensor(y_val)
        mask = [1] * len(x_train) + [0] * len(x_unlabeled)
        unlabeled_loader = DataLoader(x_unlabeled, batch_size=batch_size + 1 if len(x_unlabeled) % batch_size == 1 else batch_size, shuffle=False)
        val_loader = DataLoader(x_val, batch_size=batch_size + 1 if len(x_val) % batch_size == 1 else batch_size, shuffle=False)
        test_loader = DataLoader(x_test, batch_size=batch_size + 1 if len(x_val) % batch_size == 1 else batch_size, shuffle=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience, factor=0.6, min_lr=min_lr)

        for it in t:
            if save_prob and it % (val_freq * 5) == 0:
                self.model.eval()
                self.critic.eval()
                save_loader = graphs_to_loader_mask(x_train + x_unlabeled, y_train + y_unlabeled, mask,
                                                    batch_size=batch_size + 1 if (len(x_train) + len(x_unlabeled)) % batch_size == 1 else batch_size, shuffle=False)
                ys, gt, ps, ms = [], [], [], []
                for batch in save_loader:
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        y_hat, loss_A = self.one_batch(batch, classification)
                    p = self.critic(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, y_hat.detach().clone(), loss_A.detach().clone())
                    mask_target = batch.mask
                    ys.append(y_hat.detach().clone())
                    ps.append(p.detach().clone())
                    gt.append(batch.y.detach().clone())
                    ms.append(mask_target.detach().clone())
                ys, ps, gt, ms = torch.cat(ys).squeeze(-1) * self.mad + self.mean, torch.cat(ps).squeeze(-1), torch.cat(gt).squeeze(-1), torch.cat(ms)
                y_true, y_false = ys[ms == 1], ys[ms == 0]
                gt_true, gt_false = gt[ms == 1], gt[ms == 0]
                p_true, p_false = ps[ms == 1], ps[ms == 0]
                with open(f"unlabel_distribution_train{len(x_train)}_{it}", "wb") as fp:
                    pickle.dump([y_true.cpu().numpy(), y_false.cpu().numpy(), gt_true.cpu().numpy(), gt_false.cpu().numpy()], fp)
                with open(f"prob_distribution_{it}", "wb") as fp:
                    pickle.dump([p_true.cpu().numpy(), p_false.cpu().numpy()], fp)

            ############################################################
            ### get pseudo-labels
            ############################################################
            if it % update_freq == 0:
                if self.teacher_model is not None and it == 0:
                    print('Predicting pseudo-labels by teacher model...')
                    predict_labels = self.predict(unlabeled_loader, model=self.teacher_model)
                elif self.teacher_model is None:
                    predict_labels = self.predict(unlabeled_loader)

                if self.teacher_model is None or (self.teacher_model is not None and it == 0):
                    if classification and soft_label:
                        predict_labels = torch.sigmoid(predict_labels)
                    elif classification:
                        predict_labels = (predict_labels > 0).float()
                    train_loader = inf_iterator(graphs_to_loader_mask(x_train + x_unlabeled, y_train + predict_labels.tolist(), mask, batch_size=batch_size, shuffle=True))

            ############################################################
            ###  pretrain critic
            ############################################################
            if it == 0:
                print('Pretraining the critic.')
                self.model.eval()
                self.critic.train()
                t_critic = tqdm(range(critic_iters))
                for it_critic in t_critic:
                    batch = next(train_loader).to(self.device)
                    with torch.no_grad():
                        y_hat, loss_A = self.one_batch(batch, classification)
                    if loss_A.mean() < 0:
                        print(it_critic, y_hat.max(), y_hat.min(), torch.max(batch.y), torch.min(batch.y), loss_A.mean())
                        raise ValueError
                    p = self.critic(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, y_hat.detach().clone(), loss_A.detach().clone())
                    mask_target = batch.mask
                    if self.model.num_class > 1:
                        mask_target = mask_target.unsqueeze(dim=1).repeat(1, self.model.num_class)
                    loss_B = F.binary_cross_entropy(p, mask_target.unsqueeze(dim=-1).float(), reduction='none')
                    weight = torch.ones_like(loss_B)
                    weight[batch.mask > 0] = 10.0  # weighted BCE
                    loss_B = (loss_B * weight).sum()

                    self.optimizer.zero_grad()
                    loss_B.backward()
                    self.optimizer.step()
                    t_critic.set_description(f'Critic Iter.: {it_critic + 1} | Loss_B: {float(loss_B):.3f} | Lr(1e-4): {self.optimizer.param_groups[0]["lr"] * 1e4:.2f}')
                torch.cuda.empty_cache()

            ############################################################
            ###  start semi-supervised training
            ############################################################
            self.critic.train()
            batch = next(train_loader).to(self.device)

            ############################################################
            ### compute loss for critic
            ############################################################
            self.model.eval()
            with torch.no_grad():
                y_hat, loss_A = self.one_batch(batch, classification)
            if loss_A.mean() < 0:
                print(it, y_hat[0], batch.y[0], loss_A.mean())
                raise ValueError
            p = self.critic(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, y_hat.detach().clone(), loss_A.clone().detach())
            soft_mask = batch.mask.clone().float()
            soft_mask[soft_mask > 0.9] = 0.9
            soft_mask[soft_mask < 0.1] = 0.1
            if self.model.num_class > 1:
                soft_mask = soft_mask.unsqueeze(dim=1).repeat(1, self.model.num_class)
            loss_B = F.binary_cross_entropy(p, soft_mask.unsqueeze(dim=-1), reduction='none') * critic_loss_weight
            weight = torch.ones_like(loss_B)
            weight[batch.mask > 0] = 10.0  # weighted BCE
            loss_B = (loss_B * weight).sum()

            ############################################################
            ### compute loss for model
            ############################################################
            self.model.train()
            y_hat, loss_A = self.one_batch(batch, classification)
            p1 = p.detach().clone().squeeze()
            sample_weights = torch.zeros_like(p1, dtype=torch.float, device=self.device)
            sample_weights[batch.mask > 0.5] = (1 + labeled_weight * p1[batch.mask > 0.5]) / (batch.mask > 0.5).sum()
            if only_positive:
                p1[p1 < 0.5] = 0.5  # only use 'positive' samples, avoiding volatility
            sample_weights[batch.mask < 0.5] = unlabeled_weight * (2 * p1[batch.mask < 0.5] - 1) / (batch.mask < 0.5).sum()  # TODO: increase unlabeled_weight gradually
            loss_C = (sample_weights * loss_A).sum() / sample_weights.sum()  # a negative loss usually means loss explosion

            self.optimizer.zero_grad()
            loss = loss_B + loss_C
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)  # clip gradient to avoid the gradient explosion
            loss.backward()
            self.optimizer.step()
            t.set_description(f"Iter. {it + 1} | Loss_B: {float(loss_B):.3f} | Loss_C: {float(loss_C):.3f} | Lr(1e-4): {self.optimizer.param_groups[0]['lr'] * 1e4:.2f}")

            ############################################################
            ### evaluation on validation and test sets
            ############################################################
            if it % val_freq == 0:
                val_pred = self.predict(val_loader)

                if classification:
                    val_score = self.classification_test(val_pred, y_val.tolist(), self.model.num_class)
                    self.val_roc.append(val_score)
                else:
                    val_score = self.loss_fn(squeeze_if_needed(val_pred), y_val)
                    self.val_losses.append(val_score)
                if (classification and val_score >= max(self.val_roc)) or (not classification and val_score <= min(self.val_losses)):
                    torch.save({'config': self.config, 'model': self.model.state_dict(), 'critic': self.critic.state_dict(), 'mean': self.mean, 'mad': self.mad}, self.best_path)
                    y_hat = self.predict(test_loader)
                    if classification:
                        test_score = self.classification_test(y_hat, y_test, self.model.num_class)
                    else:
                        test_score = calc_rmse(y_test, y_hat)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(val_score)
                tqdm.write(f'[{it}] Val {val_score:.3f} | Best_Val {max(self.val_roc) if classification else min(self.val_losses):.3f} | Test: {test_score:.3f}')

        torch.save(self.val_losses, loss_path + '.pt')  # save loss weight
        plot_loss(self.val_losses, loss_path + '.pdf')  # plot the loss change
        return test_score

    def one_batch(self, batch, classification):
        y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
        if classification:
            y_hat = torch.sigmoid(y_hat)
            loss_A = F.binary_cross_entropy(squeeze_if_needed(y_hat), squeeze_if_needed(batch.y.reshape(-1, self.model.num_class)), reduction='none')
        else:
            loss_A = F.mse_loss(squeeze_if_needed(y_hat), squeeze_if_needed((batch.y - self.mean) / self.mad), reduction='none')
        return y_hat, loss_A

    def classification_test(self, preds, targets, num_class, ):
        if num_class > 1:
            rocauc_list = []
            targets = np.array(targets)
            for i in range(num_class):
                if 0 in targets[:, i] and 1 in targets[:, i]:  # AUC is only defined when there are two classes.
                    valids = (targets[:, i] != 0.5)
                    rocauc_list.append(sklearn.metrics.roc_auc_score(targets[valids, i], torch.sigmoid(preds[valids, i]).cpu().numpy()))
            test_score = sum(rocauc_list) / len(rocauc_list)
        else:
            test_score = sklearn.metrics.roc_auc_score(targets, torch.sigmoid(preds).cpu().numpy())
        return test_score

    def predict(self, loader, model=None):
        if model is None:
            model = self.model
        y_pred = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                y_hat = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).detach() * self.mad + self.mean
                y_hat = squeeze_if_needed(y_hat).tolist()
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)
        return torch.tensor(y_pred)

    def visual(self, loader):
        y_pred, feats = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                y_hat, feat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, return_feat=True)
                y_hat = squeeze_if_needed(y_hat.detach() * self.mad + self.mean).tolist()
                feats.append(feat.detach())
                if type(y_hat) is list:
                    y_pred.extend(y_hat)
                else:
                    y_pred.append(y_hat)
        return torch.tensor(y_pred), torch.cat(feats)


def graphs_to_loader(x: List[Data], y: List[float], batch_size=64, shuffle: bool = False):
    for graph, label in zip(x, y):
        graph.y = torch.tensor(label)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)


def graphs_to_loader_mask(x: List[Data], y: List[float], masks: List[int], batch_size=64, shuffle: bool = False):
    for graph, label, mask in zip(x, y, masks):
        graph.y = torch.tensor(label)
        graph.mask = torch.tensor(mask)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle, drop_last=False)  # drop_last=True to prevent BatchNorm error


def squeeze_if_needed(tensor):
    from torch import Tensor
    if len(tensor.shape) > 1 and tensor.shape[1] == 1 and type(tensor) is Tensor:
        tensor = tensor.squeeze()
    return tensor


def plot_loss(losses, save_path="loss.pdf"):
    epochs = [i for i in range(len(losses))]
    plt.plot(epochs, losses)
    plt.savefig(save_path)
