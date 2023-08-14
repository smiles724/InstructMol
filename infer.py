import argparse

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from benchmark import UnlabeledData, Descriptors
from models import squeeze_if_needed, get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/ace.yml')
    parser.add_argument('--data', metavar='DATA', type=str, default='CHEMBL214_Ki')
    parser.add_argument('--model', metavar='MODEL', type=str, default='GIN', choices=['MPNN', 'GAT', 'GCN', 'GIN'])

    parser.add_argument('--target_path', type=str, default='./CHEMBL_214')   # ./250k_zinc
    parser.add_argument('--model_path', type=str, default='./results/GIN_CHEMBL214_Ki_best_0.pt')
    args = parser.parse_args()
    algorithm = get_model(args.model)
    ckpt = torch.load(args.model_path)
    config = ckpt['config']

    target_data = UnlabeledData(args.target_path, size_limit=-1)
    if target_data.x is None:
        target_data(Descriptors.GRAPH, save_mode=False)
    x_target = target_data.x
    loader = DataLoader(x_target, batch_size=config.train.batch + 1 if len(x_target) % config.train.batch == 1 else config.train.batch, shuffle=False)
    print(f'Model weight: {args.model_path} | Unlabeled Size: {len(x_target)}')

    algorithm = get_model(args.model)

    model = algorithm(mean=ckpt['mean'], mad=ckpt['mad'], config=config, **config.model)
    model.model.load_state_dict(ckpt['model'])
    model.critic.load_state_dict(ckpt['critic'])

    y_pred, p_pred = [], []
    model.model.eval()
    model.critic.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(model.device)
            y_hat = model.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch).detach() * ckpt['mad'] + ckpt['mean']

            # assume some sort of loss
            p = model.critic(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch, y_hat.detach().clone(), torch.ones_like(y_hat).squeeze(-1))

            y_hat = squeeze_if_needed(y_hat).tolist()
            p = squeeze_if_needed(p).tolist()
            if type(y_hat) is list:
                y_pred.extend(y_hat)
            else:
                y_pred.append(y_hat)
            if type(p) is list:
                p_pred.extend(p)
            else:
                p_pred.append(p)

    df = pd.DataFrame({'smiles': target_data.smiles, 'Ki': [10 ** (-x) / 5 for x in y_pred], 'confidence': p_pred})  #
    df.to_csv(f'./{args.target_path}_inference.csv')
