import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

import Data
from benchmark import ace_datasets, moleculenet_reg, moleculenet_cls, moleculenet_tasks, Descriptors
from models import get_model, calc_rmse

parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DATA', type=str, default='CHEMBL214_Ki', choices=ace_datasets + moleculenet_reg + moleculenet_cls)
parser.add_argument('--normalize', type=bool, default=True, help='scale the label between 0 and 1')
parser.add_argument('--model', metavar='MODEL', type=str, default='GIN', choices=['MPNN', 'GAT', 'GCN', 'GIN'])
parser.add_argument('--aggr', type=str, default='transformer', choices=['pool', 'transformer'], help='aggregation function to get graph-level features')
parser.add_argument('--batch', metavar='SIZE', type=int, default=64, help='batch size')
parser.add_argument('--path', metavar='PATH', type=str, default='')
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--unlabeled_path', metavar='PRE', type=str, default='./250k_zinc')
parser.add_argument('--unlabeled_size', type=int, default=-1, help='how many samples used for the entire instruct learning, -1 means not limitation.')
args = parser.parse_args()
algorithm = get_model(args.model)

print(f'LabeledData: {args.data}')
data = Data(args.data)
data(Descriptors.GRAPH)  # GNNs
if args.data not in moleculenet_cls and args.normalize:
    data.compute_mean_mad()
hyper = {'dropout': 0.4, 'edge_hidden': 256, 'fc_hidden': 512, 'num_layers': 3, 'n_fc_layers': 1, 'node_hidden': 128, 'transformer_hidden': 128, 'aggr': args.aggr}
if args.data in moleculenet_tasks.keys():
    hyper['num_class'] = moleculenet_tasks[args.data]

x_train, y_train, x_val, y_val, x_test, y_test = data.x_train, data.y_train, data.x_val, data.y_val, data.x_test, data.y_test
if args.data in moleculenet_cls:
    mean, mad = 0.0, 1.0
    classification = True
else:
    mean, mad = data.mean, data.mad
    classification = False
model = algorithm(**hyper)
if not args.path:
    args.path = f'results/GIN_{args.data}_best.pt'
print(f'Loading pretrained weight from {args.path}.')
weight = torch.load(args.path)
if type(weight) is List:
    weight = weight[0]
model.model.load_state_dict(weight)

train_loader = DataLoader(x_train, batch_size=args.batch + 1 if len(x_train) % args.batch == 1 else args.batch, shuffle=False)
y_hat_train, feats_train = model.visual(train_loader, mean=mean, mad=mad)
test_loader = DataLoader(x_test, batch_size=args.batch + 1 if len(x_train) % args.batch == 1 else args.batch, shuffle=False)
y_hat_test, feats_test = model.visual(test_loader, mean=mean, mad=mad)
feats = torch.cat([feats_train, feats_test])
y = y_train + y_test
y_hat = torch.cat([y_hat_train, y_hat_test])
score = calc_rmse(y, y_hat)
print(f'RMSE: {score:.3f}')

for iter in [250, 500, 1000, 2000]:
    tsne = TSNE(n_components=2, verbose=1, random_state=2023, n_iter=iter)
    z = tsne.fit_transform(feats.cpu().numpy())
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=1.5, c=np.array(y), alpha=0.8)   # bad using 10 ** (-np.array(y))
    plt.colorbar(label='property')
    plt.savefig(f'./tsne_{args.data}_{iter}.pdf', bbox_inches='tight')




