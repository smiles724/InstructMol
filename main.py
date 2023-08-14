import os
import copy
import argparse
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

import torch
from torch_geometric.loader import DataLoader
from models import get_model, calc_rmse
from utils import set_seed, Logger, load_config
from benchmark import ace_datasets, moleculenet_reg, moleculenet_cls, UnlabeledData, moleculenet_tasks, Descriptors, LabeledData


def cross_validate(n_folds):
    x_train, y_train, x_test, y_test, mean, mad = data.x_train, data.y_train, data.x_test, data.y_test, data.mean, data.mad
    test_loader = DataLoader(x_test, batch_size=batch_size + 1 if len(x_test) % batch_size == 1 else batch_size, shuffle=False)

    ss = StratifiedKFold(n_splits=n_folds, shuffle=True)
    labels = [0 if i < np.median(y_train) else 1 for i in y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    scores_before, scores = [], []
    log.logger.info(f'{"=" * 35} {args.data} {"=" * 35}\nModel: {args.model}, Norm: {bool(config.data.normalize)}, Batch: {batch_size}\n'
                    f'Train/Val/Test/Unlabeled: {int(len(y_train) * (1 - 1 / n_folds))}/{int(len(y_train) / n_folds)}/{len(y_test)}/{len(x_unlabeled)}')
    for i_split, split in enumerate(splits):
        x_tr_fold, y_tr_fold = [copy.deepcopy(x_train[i]) for i in split['train_idx']], [copy.deepcopy(y_train[i]) for i in split['train_idx']]
        x_val_fold, y_val_fold = [copy.deepcopy(x_train[i]) for i in split['val_idx']], [copy.deepcopy(y_train[i]) for i in split['val_idx']]

        model = algorithm(loss_fn=loss_fn, best_path=best_path + f'_{i_split}.pt', mean=mean, mad=mad, config=config, **config.model)
        print(f'{"=" * 30} Start Training [Fold {i_split + 1}/{n_folds}] {"=" * 33}')
        if not os.path.exists(base_path + f'_{i_split}.pt'):
            print('Start pretraining the model.')
            model.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, config.train.epochs, batch_size, base_path=base_path + f'_{i_split}.pt')
        print(f'Loading weight from {base_path + f"_{i_split}.pt"}.')
        weight, val_last = torch.load(base_path + f'_{i_split}.pt')
        model.model.load_state_dict(weight)
        model.val_losses = [val_last]

        y_hat = model.predict(test_loader)
        score_before = calc_rmse(y_test, y_hat)
        print(f"Before instruct learning -- rmse: {score_before:.3f}")

        score = model.InstructLearning(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, x_unlabeled, x_test, y_test, batch_size=batch_size, loss_path=loss_path,
                                       **config.train.instruct_learning)
        scores_before.append(score_before)
        scores.append(score)
        if args.debug: break
    return sum(scores_before) / len(scores_before), sum(scores) / len(scores)


def molnet_train():
    x_train, y_train, x_val, y_val, x_test, y_test = data.x_train, data.y_train, data.x_val, data.y_val, data.x_test, data.y_test
    if args.data in moleculenet_cls:
        classification = True
        mean, mad = 0.0, 1.0
    else:
        mean, mad = data.mean, data.mad
        classification = False
    log.logger.info(f'{"=" * 35} {args.data} TASKS {"=" * 35}\nModel: {args.model}, Norm: {bool(config.data.normalize)}, '
                    f'Train/Val/Test/Unlabeled: {len(y_train)}/{len(y_val)}/{len(y_test)}/{len(x_unlabeled)}\nBatch: {batch_size},'
                    f' CLS: {classification}\n{"=" * 30} Start Training {"=" * 33}')
    model = algorithm(num_class=moleculenet_tasks.get(args.data, 1), loss_fn=loss_fn, best_path=best_path, mean=mean, mad=mad, config=config, **config.model)
    if not os.path.exists(base_path):
        print('Start pretraining the model.')
        model.train(x_train, y_train, x_val, y_val, config.train.epochs, batch_size, base_path=base_path, classification=classification)
    print(f'Loading pretrained weight from {base_path}.')
    weight, val_last = torch.load(base_path)
    model.model.load_state_dict(weight)
    model.val_losses = [val_last]

    test_loader = DataLoader(x_test, batch_size=batch_size + 1 if len(x_test) % batch_size == 1 else batch_size, shuffle=False)
    y_hat = model.predict(test_loader)
    if args.data in moleculenet_cls:
        if model.model.num_class > 1:
            rocauc_list = []
            y_test = np.array(y_test)
            for i in range(model.model.num_class):
                if 0 in y_test[:, i] and 1 in y_test[:, i]:  # AUC is only defined when there are two classes.
                    valids = (y_test[:, i] != 0.5)
                    rocauc_list.append(sklearn.metrics.roc_auc_score(y_test[valids, i], torch.sigmoid(y_hat[valids, i]).cpu().numpy()))
            score_before = sum(rocauc_list) / len(rocauc_list)
        else:
            score_before = sklearn.metrics.roc_auc_score(data.y_test, torch.sigmoid(y_hat).cpu().numpy())  # sigmoid to 0 - 1
        print(f"Before instruct learning -- roc-auc: {score_before:.3f}")
    else:
        score_before = calc_rmse(y_test, y_hat)
        print(f"Before instruct learning -- rmse: {score_before:.3f}")

    score = model.InstructLearning(x_train, y_train, x_val, y_val, x_unlabeled, x_test, y_test, batch_size=batch_size, loss_path=loss_path,
                                   classification=classification, **config.train.instruct_learning)
    return score_before, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--data', metavar='DATA', type=str, default='CHEMBL214_Ki', choices=ace_datasets + moleculenet_reg + moleculenet_cls + ['logp'])
    parser.add_argument('--model', metavar='MODEL', type=str, default='MPNN', choices=['MPNN', 'GAT', 'GCN', 'GIN'])
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    config, config_name = load_config(args.config)
    batch_size = config.train.batch
    set_seed(config.train.seed)
    log = Logger('logs/', f'train_{args.model}_{args.data}.log')
    os.makedirs(config.train.result_dir, exist_ok=True)
    loss_path = config.train.result_dir + f'{args.model}_{args.data}_loss'
    best_path = config.train.result_dir + f'{args.model}_{args.data}_best'
    base_path = config.train.result_dir + f'{args.model}_{args.data}_base'
    algorithm = get_model(args.model)
    print(config)

    try:
        data = LabeledData(args.data)
        data(Descriptors.GRAPH)  # we use GNNs
        if args.data not in moleculenet_cls and config.data.normalize:
            data.compute_mean_mad()
        unlabeled_data = UnlabeledData(config.data.unlabeled_path, config.data.unlabeled_size, logp=True if args.data == 'logp' else False)
        if unlabeled_data.x is None:
            unlabeled_data(Descriptors.GRAPH)
        x_unlabeled = unlabeled_data.x
        loss_fn = torch.nn.MSELoss()
        if args.data in ['tox21', 'toxcast']:  # nan value
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif args.data in moleculenet_cls:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        if args.data in ace_datasets:
            metric_before, metric = cross_validate(config.train.n_folds)
        else:
            best_path = best_path + '.pt'
            base_path = base_path + '.pt'
            metric_before, metric = molnet_train()
        log.logger.info(f'{args.data}: {metric_before:.3f} --> {metric:.3f}. ')
    except KeyboardInterrupt:
        log.logger.info(f'Stop training for {args.data}')
