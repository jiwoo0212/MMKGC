# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import numpy as np
import torch
import importlib
import datetime
import random
import os


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)
    
# visualizing
import re
import matplotlib.pyplot as plt

def parse_log_file(log_file):
    epochs = []
    recall_at_20_train = []
    recall_at_20_val = []
    ndcg_at_20_train = []
    ndcg_at_20_val = []
    
    epoch_pattern = re.compile(r'INFO epoch (\d+) training')
    recall_pattern = re.compile(r'recall@20: ([0-9\.]+)')
    ndcg_pattern = re.compile(r'ndcg@20: ([0-9\.]+)')
    
    with open(log_file, 'r') as file:
        lines = file.readlines()
        current_epoch = None
        for i, line in enumerate(lines):
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)
            
            if 'valid result' in line:
                recall_match = recall_pattern.search(lines[i+1])
                ndcg_match = ndcg_pattern.search(lines[i+1])
                if recall_match and ndcg_match:
                    recall_at_20_val.append(float(recall_match.group(1)))
                    ndcg_at_20_val.append(float(ndcg_match.group(1)))
            
            if 'test result' in line:
                recall_match = recall_pattern.search(lines[i+1])
                ndcg_match = ndcg_pattern.search(lines[i+1])
                if recall_match and ndcg_match:
                    recall_at_20_train.append(float(recall_match.group(1)))
                    ndcg_at_20_train.append(float(ndcg_match.group(1)))
    
    return epochs, recall_at_20_train, recall_at_20_val, ndcg_at_20_train, ndcg_at_20_val


def plot_metrics(log_file_path, fig_path):
    log_name = log_file_path.split('/')[-1][:-4]
    # print(len(epochs), len(recall_val), len(recall_test), len(ndcg_val), len(ndcg_test))
    epochs, recall_val, recall_test, ndcg_val, ndcg_test = parse_log_file(log_file_path)
    recall_test = recall_test[:-1]
    ndcg_test = ndcg_test[:-1]

    max_recall_val = max(recall_val, default=0)
    max_recall_test = max(recall_test, default=0)
    max_ndcg_val = max(ndcg_val, default=0)
    max_ndcg_test = max(ndcg_test, default=0)
    
    best_epoch_recall_test = epochs[recall_test.index(max_recall_test)] if recall_test else None
    best_epoch_ndcg_test = epochs[ndcg_test.index(max_ndcg_test)] if ndcg_test else None
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'{log_name}', fontsize=14,) #fontweight='bold')
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, recall_val, label='Val Recall@20', marker='o')
    plt.plot(epochs, recall_test, label='Test Recall@20', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.legend()
    plt.title(f'val/test Recall@20 (Max test: {max_recall_test:.4f}, Max val: {max_recall_val:.4f} at Epoch {best_epoch_recall_test})')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ndcg_val, label='Val NDCG@20', marker='o')
    plt.plot(epochs, ndcg_test, label='Test NDCG@20', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('NDCG@20')
    plt.legend()
    plt.title(f'val/test NDCG@20 (Max test: {max_ndcg_test:.4f}, Max val: {max_ndcg_val:.4f} at Epoch {best_epoch_ndcg_test})')    
    plt.tight_layout()
    plt.savefig(fig_path)

