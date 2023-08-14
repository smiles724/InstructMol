import argparse
import os
import copy

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from benchmark import ace_datasets, moleculenet_reg, moleculenet_cls, UnlabeledData, Descriptors, LabeledData
from models import get_model, calc_rmse
from utils import set_seed, Logger, load_config


def cross_validate(n_folds):
    x_train, y_train, x_test, y_test= data.x_train, data.y_train, data.x_test, data.y_test
    test_loader = DataLoader(x_test, batch_size=batch_size + 1 if len(x_test) % batch_size == 1 else batch_size, shuffle=False)

    ss = StratifiedKFold(n_splits=n_folds, shuffle=True)
    labels = [0 if i < np.median(y_train) else 1 for i in y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    scores_before, scores = [], [0] * len(splits)
    log.logger.info(f'{"=" * 35} {args.data} {"=" * 35}\nModel: {args.model}, Norm: {bool(config.data.normalize)}, Batch: {batch_size}\n'
                    f'Train/Val/Test/Unlabeled: {int(len(y_train) * (1 - 1 / n_folds))}/{int(len(y_train) / n_folds)}/{len(y_test)}/{len(x_unlabeled)}')

    for i_split, split in enumerate(splits):
        x_tr_fold, y_tr_fold = [copy.deepcopy(x_train[i]) for i in split['train_idx']], [copy.deepcopy(y_train[i]) for i in split['train_idx']]
        x_val_fold, y_val_fold = [copy.deepcopy(x_train[i]) for i in split['val_idx']], [copy.deepcopy(y_train[i]) for i in split['val_idx']]

        model = algorithm(loss_fn=loss_fn, best_path=best_path + f'_{i_split}.pt', mean=data.mean, mad=data.mad, config=config, **config.model)

        if config.teacher.type == 'GIN':  # TODO: other types of model
            from models.gin import GINVirtual_node
            model.teacher_model = GINVirtual_node(aggr='transformer')
            ckpt = torch.load(config.teacher.path)
            model.teacher_model.load_state_dict(ckpt['model'])
            model.teacher_model.to(model.device)
            print(f'Loading teacher model from {config.teacher.path} successfully.')

        print(f'{"=" * 30} Start Training [Fold {i_split + 1}/{n_folds}] {"=" * 33}')
        if config.teacher.stage == 'supervised':
            print('Start training the model with a teacher model.')
            unlabeled_loader = DataLoader(x_unlabeled, batch_size=batch_size + 1 if len(x_unlabeled) % batch_size == 1 else batch_size, shuffle=False)
            predict_labels = model.predict(unlabeled_loader, model=model.teacher_model)

            model.train(x_tr_fold + x_unlabeled, y_tr_fold + predict_labels.tolist(), x_val_fold, y_val_fold, config.train.epochs, batch_size, base_path=base_path + '_student.pt')

        elif not os.path.exists(base_path + f'_{i_split}.pt'):
            print('Start pretraining the model.')
            model.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, config.train.epochs, batch_size, base_path=base_path + f'_{i_split}.pt')
        weight, val_last = torch.load(base_path)
        model.model.load_state_dict(weight)
        model.val_losses = [val_last]

        y_hat = model.predict(test_loader)
        score_before = calc_rmse(y_test, y_hat)
        scores_before.append(score_before)
        if config.teacher.stage == 'supervised': continue
        print(f"Before instruct learning -- rmse: {score_before:.3f}")
        score = model.InstructLearning(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, x_unlabeled, x_test, y_test, batch_size=batch_size, loss_path=loss_path,
                                       **config.train.instruct_learning)
        scores[i_split] = score
        if args.debug: break
    return sum(scores_before) / len(scores_before), sum(scores) / len(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--data', metavar='DATA', type=str, default='CHEMBL214_Ki', choices=ace_datasets + moleculenet_reg + moleculenet_cls)
    parser.add_argument('--model', metavar='MODEL', type=str, default='MPNN', choices=['MPNN', 'GAT', 'GCN', 'GIN'])
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    config, config_name = load_config(args.config)
    batch_size = config.train.batch
    set_seed(config.train.seed)
    log = Logger('logs/', f's_{args.model}{"_t_" + config.teacher.type}_{args.data}.log')
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
        unlabeled_data = UnlabeledData(config.data.unlabeled_path, config.data.unlabeled_size)
        if unlabeled_data.x is None:
            unlabeled_data(Descriptors.GRAPH)
        x_unlabeled = unlabeled_data.x
        loss_fn = torch.nn.MSELoss()

        metric_before, metric = cross_validate(config.train.n_folds)
        print(f'{args.data}: {metric_before:.3f} --> {metric:.3f}.')
    except KeyboardInterrupt:
        print(f'Stop training for {args.data}')
