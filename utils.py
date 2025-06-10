from typing import Tuple

import dgl
import torch
from dgl import load_graphs
from torch_geometric.datasets import Planetoid, Actor, WebKB, WikipediaNetwork, HeterophilousGraphDataset
from torch_geometric.utils import coalesce, from_dgl

import torch.nn.functional as F


def load_dataset(dataset_name, device='cuda:0'):
    if dataset_name in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root='data/', name=dataset_name)
        data = dataset[0].to(device)
    elif dataset_name in ['squirrel', 'chameleon']:
        dataset = WikipediaNetwork(root='data/', geom_gcn_preprocess=True, name=dataset_name)
        data = dataset[0].to(device)
    elif dataset_name == 'actor':
        dataset = Actor(root='data/Actor')
        data = dataset[0].to(device)
    elif dataset_name in ['minesweeper', 'questions', 'roman_empire', 'tolokers', 'amazon_ratings']:
        dataset = HeterophilousGraphDataset(root='data/', name=dataset_name)
        data = dataset[0].to(device)
    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=dataset_name, split='geom-gcn')
        data = dataset[0].to(device)
    else:
        raise NotImplementedError

    return dataset, data


table_datasets = [
    'actor', 'chameleonf', 'squirrelf',
    'romanempire', 'amazonratings',
    'blogcatalog', 'flickr',
    'photo', 'wikics', 'pubmed',
]


def load_data(
        dataset_name: str,
        normalize: int = -1,
        undirected: bool = True,
        self_loop: bool = True,
):
    dataset_name = dataset_name.lower()
    if dataset_name in table_datasets:
        print("Load Dataset: ", dataset_name)
        file_path = f'new_data/graphs/{dataset_name}.pt'
        graphs, _ = load_graphs(file_path)
        graph = graphs[0]
        label = graph.ndata['label']
        graph.ndata['y'] = graph.ndata['label']
        class_num = (torch.max(label) + 1).long().item()
        if normalize != -1:
            graph.ndata['x'] = F.normalize(graph.ndata['feat'], dim=1, p=normalize)
        else:
            graph.ndata['x'] = graph.ndata['feat']
        if undirected:
            graph = dgl.to_bidirected(graph, copy_ndata=True)
        if self_loop:
            graph = graph.remove_self_loop().add_self_loop()
        data = from_dgl(graph)
    else:
        raise NotImplementedError
    return data, label, class_num


def load_fixed_data_split(dataname, split_idx, device='cuda:0'):
    """load fixed split for benchmark dataset, train/val/test is 48%/32%/20%.
    Parameters
    ----------
    dataname: str
        dataset name.
    split_idx: int
        id of split plan.
    """
    splits_file_path = f'./new_data/splits/{dataname}_splits.pt'
    splits_file = torch.load(splits_file_path)
    train_mask_list = splits_file['train']
    val_mask_list = splits_file['val']
    test_mask_list = splits_file['test']

    # 放到对应device上
    train_mask = torch.BoolTensor(train_mask_list[split_idx]).to(device)
    val_mask = torch.BoolTensor(val_mask_list[split_idx]).to(device)
    test_mask = torch.BoolTensor(test_mask_list[split_idx]).to(device)
    return train_mask, val_mask, test_mask
