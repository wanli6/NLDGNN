import csv
import math
import os
import torch
from torch import nn
import numpy as np
import uuid
import warnings
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, remove_self_loops, degree

from models import NLDGNN, GCN, MLP
from datasets import Dataset
from utils import load_data, load_fixed_data_split

warnings.filterwarnings("ignore", category=UserWarning)


class GraphTrainer:
    def __init__(self, args, device='cuda:0'):
        self.args = args
        self.device = device
        self.setup_data()
        self.initialize_models()
        self.table_dataset = [
            'actor', 'chameleonf', 'squirrelf',
            'romanempire', 'amazonratings',
            'blogcatalog', 'flickr',
            'photo', 'wikics', 'pubmed',
        ]

    def setup_data(self):
        citation = ['texas', 'wisconsin', 'cornell']
        hetero_graphs = ['minesweeper', 'questions', 'roman_empire', 'tolokers', 'amazon_ratings',
                         'squirrel_filtered_directed', 'texas_4_classes', 'chameleon_filtered_directed']
        table_datasets = [
            'actor', 'chameleonf', 'squirrelf',
            'romanempire', 'amazonratings',
            'blogcatalog', 'flickr',
            'photo', 'wikics', 'pubmed',
        ]

        if self.args.data in citation:
            dataset = WebKB(root='data/', name=self.args.data)
            self.data = dataset[0].to(self.device)
            self.num_classes = dataset.num_classes
        elif self.args.data in ['squirrel', 'chameleon']:
            dataset = WikipediaNetwork(root='data/', name=self.args.data)
            self.data = dataset[0].to(self.device)
            self.num_classes = dataset.num_classes
        elif self.args.data == 'actor':
            dataset = Actor(root='data/Actor')
            self.data = dataset[0].to(self.device)
            self.num_classes = dataset.num_classes
        elif self.args.data in hetero_graphs:
            dataset = Dataset(name=self.args.data,
                              add_self_loops=False,
                              device=self.device,
                              use_sgc_features=False,
                              use_identity_features=False,
                              use_adjacency_features=False,
                              do_not_use_original_features=False)
            x = dataset.node_features
            y = dataset.labels.long()
            src, dst = dataset.graph.edges()
            edge_index = torch.stack([src, dst], dim=0)
            self.data = Data(x=x, edge_index=edge_index, y=y)
            self.data.train_mask = dataset.train_mask.t()
            self.data.val_mask = dataset.val_mask.t()
            self.data.test_mask = dataset.test_mask.t()
            self.num_classes = len(torch.unique(y))
        elif self.args.data in table_datasets:
            self.data, y, self.num_classes = load_data(self.args.data)
            self.data = self.data.to(self.device)
        else:
            dataset = Planetoid(root='data', name=self.args.data, split='geom-gcn')
            self.data = dataset[0].to(self.device)
            self.num_classes = dataset.num_classes

        self.y = torch.nn.functional.one_hot(self.data.y).float().to(self.device)
        self.data.edge_index = add_remaining_self_loops(self.data.edge_index)[0]

        if self.args.estimator == 'gcn':
            dense_adj = to_dense_adj(self.data.edge_index)[0]
            deg = dense_adj.sum(dim=1, keepdim=True)
            deg = deg.pow(-0.5)
            dense_adj = deg.view(-1, 1) * dense_adj * deg.view(1, -1)
            self.sparse_adj = dense_adj.to_sparse()

    def initialize_models(self):
        if self.args.estimator == 'mlp':
            self.predictor = MLP(self.data.num_features, 64, self.num_classes, self.args.dropout).to(self.device)
        else:
            self.predictor = GCN(self.data.num_features, 64, self.num_classes, self.args.dropout).to(self.device)

        if self.args.data in ['cora', 'citeseer', 'pubmed']:
            init_v = 'gcn'
        else:
            init_v = 'mlp'

        self.model = NLDGNN(self.data.num_features, self.args.hidden, self.num_classes,
                            dropout=self.args.dropout, k=self.args.k, nlayers=self.args.nlayers,
                            gamma=self.args.gamma, alpha=self.args.alpha, beta=self.args.beta,
                            eps=self.args.eps, lamda=self.args.lamda, num_nodes=self.data.num_nodes,
                            r=self.args.r, init_v=init_v, weight=self.args.weight).to(self.device)

    def train_step(self, train_mask, model, optimizer, criterion, adj=None):
        model.train()
        optimizer.zero_grad()
        if adj is None:
            if self.args.estimator == 'mlp':
                out = model(self.data.x)
            else:
                out = model(self.data.x, self.sparse_adj)
        else:
            out = model(self.data.x, adj, self.edge_index, self.edge_weight, self.connect, self.confidence)
        loss = criterion(out[train_mask], self.data.y[train_mask])
        if adj is None:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
        optimizer.step()
        return loss

    def val_step(self, val_mask, model, criterion, adj=None):
        model.eval()
        with torch.no_grad():
            if adj is None:
                if self.args.estimator == 'mlp':
                    out = model(self.data.x)
                else:
                    out = model(self.data.x, self.sparse_adj)
            else:
                out = model(self.data.x, adj, self.edge_index, self.edge_weight, self.connect, self.confidence)
            pred = out.argmax(dim=1)
            loss = criterion(out[val_mask], self.data.y[val_mask])
            acc = int((pred[val_mask] == self.data.y[val_mask]).sum()) / int(val_mask.sum())
            return loss.item(), acc, out

    def test_step(self, test_mask, model, criterion, adj=None):
        model.eval()
        with torch.no_grad():
            if adj is None:
                if self.args.estimator == 'mlp':
                    out = model(self.data.x)
                else:
                    out = model(self.data.x, self.sparse_adj)
            else:
                out = model(self.data.x, adj, self.edge_index, self.edge_weight, self.connect, self.confidence)
            pred = out.argmax(dim=1)
            loss = criterion(out[test_mask], self.data.y[test_mask])
            acc = int((pred[test_mask] == self.data.y[test_mask]).sum()) / int(test_mask.sum())
            return loss.item(), acc

    def train_mlp(self, train_mask, val_mask, test_mask):
        mlp_checkpt_file = 'trained_model_dict/' + uuid.uuid4().hex + 'mlp.pt'
        criterion_mlp = nn.CrossEntropyLoss()
        mlp_optimizer = torch.optim.Adam(params=self.predictor.parameters(),
                                         lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        best = 100
        patience = self.args.patience
        count = 0

        for i in range(1000):
            _ = self.train_step(train_mask, self.predictor, mlp_optimizer, criterion_mlp)
            val_loss, val_acc, _ = self.val_step(val_mask, self.predictor, criterion_mlp)
            if val_loss < best:
                count = 0
                best = val_loss
                torch.save(self.predictor.state_dict(), mlp_checkpt_file)
            else:
                count += 1
                if count == patience:
                    break

        self.predictor.load_state_dict(torch.load(mlp_checkpt_file))
        _, best_acc = self.test_step(test_mask, self.predictor, criterion_mlp)
        return best_acc

    def run_training(self, split=0):
        # Setup masks
        if self.args.data in self.table_dataset:
            train_mask, val_mask, test_mask = load_fixed_data_split(self.args.data, split, self.device)
        else:
            train_mask = self.data.train_mask[:, split]
            val_mask = self.data.val_mask[:, split]
            test_mask = self.data.test_mask[:, split]

        # Train MLP/GCN predictor
        mlp_acc = self.train_mlp(train_mask, val_mask, test_mask)
        # print(f'{self.args.estimator} acc: {mlp_acc:.3f}')

        # Get pseudo labels
        if self.args.estimator == 'mlp':
            mlp_out = self.predictor(self.data.x)
        else:
            mlp_out = self.predictor(self.data.x, self.sparse_adj)

        pseudo_labels = mlp_out.softmax(dim=1)
        mask = torch.where(train_mask == True, torch.Tensor([1.]).to(self.device),
                           torch.Tensor([0.]).to(self.device)).unsqueeze(-1)
        pseudo_labels = mask * self.y + (1 - mask) * pseudo_labels

        # Prepare adjacency matrix
        self.edge_index, _ = remove_self_loops(self.data.edge_index)
        row, col = self.edge_index
        self.connect = torch.sparse_coo_tensor(self.edge_index, torch.ones_like(self.edge_index[0]),
                                               size=[self.y.size(0), self.y.size(0)]).float()

        self.edge_weight = (pseudo_labels[row] * pseudo_labels[col]).sum(dim=1)
        self.edge_weight = self.edge_weight - self.args.delta

        confidence = torch.linalg.vector_norm(pseudo_labels, dim=1) - 1 / self.num_classes
        confidence = confidence[row] * confidence[col]
        self.confidence = torch.stack([confidence, torch.log(math.e - 1 + torch.abs(self.edge_weight))], dim=0)

        if self.args.large_scale == 1:
            deg = degree(col, num_nodes=self.data.num_nodes)
            self.edge_weight = self.edge_weight / deg[col]
            self.edge_weight[self.edge_weight < -5] = -5

        adj = torch.sparse_coo_tensor(self.edge_index, self.edge_weight, size=[self.y.size(0), self.y.size(0)])

        # Train main model
        checkpt_file = 'trained_model_dict/' + uuid.uuid4().hex + '.pt'
        criterion = nn.CrossEntropyLoss()
        gnn_optimizer = torch.optim.Adam(params=self.model.parameters(),
                                         lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        best = 100
        best_acc = 0
        patience = self.args.patience
        count = 0

        for j in range(1000):
            _ = self.train_step(train_mask, self.model, gnn_optimizer, criterion, adj=adj)
            val_loss, val_acc, out = self.val_step(val_mask, self.model, criterion, adj=adj)
            if val_loss < best:
                count = 0
                best = val_loss
                torch.save(self.model.state_dict(), checkpt_file)
            else:
                count += 1
                if count == patience:
                    break

        self.model.load_state_dict(torch.load(checkpt_file))
        _, best_acc = self.test_step(test_mask, self.model, criterion, adj=adj)
        return best_acc


def objective(trial, args):
    # Suggest hyperparameters
    args.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    args.dropout = trial.suggest_float('dropout', 0.1, 0.9)
    args.hidden = trial.suggest_categorical('hidden', [64, 256, 1024])
    args.alpha = trial.suggest_float('alpha', 0.1, 0.9)
    # args.gamma = trial.suggest_int('gamma', 1, 5)
    args.k = trial.suggest_int('k', 5, 50)
    args.delta = trial.suggest_float('delta', 0.1, 0.9)
    args.beta = trial.suggest_float('beta', 0.1, 0.9)
    args.eps = trial.suggest_categorical('eps', [0.01, 0.001, 0.005])
    args.lamda = trial.suggest_float('lamda', 0.1, 0.9)
    args.r = trial.suggest_float('r', 0.1, 0.9)
    args.weight = trial.suggest_float('weight', 0.1, 0.9)
    args.estimator = trial.suggest_categorical('estimator', ['mlp', 'gcn'])

    trainer = GraphTrainer(args)
    accuracies = []

    for split in range(10):
        acc = trainer.run_training(split)
        print(f'Split {split} acc: {acc:.3f}')
        accuracies.append(acc)
        trial.report(np.mean(accuracies), step=split)

        if trial.should_prune():
            raise optuna.TrialPruned()

    # 将结果ACC和STD保存到csv文件中
    csv_file = 'log/hypersearch.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(['Dataset', 'Acc', 'Std'])
        writer.writerow(
            [args.data, f"{np.mean(accuracies):.2f}", f"{np.std(accuracies):.2f}"])

    return np.mean(accuracies)


if __name__ == "__main__":
    import argparse
    import optuna

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping.')
    parser.add_argument('--data', type=str, default='romanempire', help='Data set to be used.')
    parser.add_argument('--estimator', type=str, default='mlp', help='Estimator to generate pseudo labels.')
    parser.add_argument('--large_scale', type=int, default=1,
                        help='If set to 1, the signed adjacency matrix will be row-normalized.')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of convolutional layers.')
    parser.add_argument('--gamma', type=int, default=1,
                        help='Number of update iterations.')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of trials')

    args = parser.parse_args()

    study_name = f"{args.data}_result"
    # 不使用明文路径，获取当前路径
    db_path = os.path.join(os.path.join(os.path.abspath('.'), "hypersearch_results"), "{}.db".format(study_name))
    # db_path = os.path.join("D:\\CodePlace\\PyCharmProjects\\NLDGNN\\hypersearch_results", "{}.db".format(study_name))
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(direction='maximize',
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")