import random
import os
import numpy as np
import math
from collections import Counter

import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_params(module):
    if isinstance(module, nn.Linear):
        stdv = 1.0 / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
        if module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
        print("Initialized Weight Parameters with Uniform Distribution")
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        print("Initialized Weight Parameters with Normal Distribution") 

def normalize_features(mx):
     rowsum = mx.sum(1)
     r_inv = torch.pow(rowsum, -1)
     r_inv[torch.isinf(r_inv)] = 0.
     r_mat_inv = torch.diag(r_inv)
     mx = r_mat_inv @ mx
     return mx

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    #print("Adjacency Matrix Normalized")
    return mx

# def normalize_adj_to_sparse_tensor(mx):
#     mx = normalize_adj(mx)
#     mx = sparse_mx_to_torch_sparse_tensor(mx)
#     sparsetensor = SparseTensor(row=mx._indices()[0], col=mx._indices()[1], value=mx._values(), sparse_sizes=mx.size()).cuda()
#     return sparsetensor

def normalize_adj_to_sparse_tensor(mx, device='cuda'):
    mx = normalize_adj(mx)
    mx = sparse_mx_to_torch_sparse_tensor(mx)
    sparsetensor = SparseTensor(
        row=mx._indices()[0], 
        col=mx._indices()[1], 
        value=mx._values(), 
        sparse_sizes=mx.size()
    ).to(device)
    return sparsetensor
    
def get_syn_eigen(real_eigenvals, real_eigenvecs, eigen_k, ratio, step=1):
    k1 = math.ceil(eigen_k * ratio)
    k2 = eigen_k - k1
    print("k1:", k1, ",", "k2:", k2)
    k1_end = (k1 - 1) * step + 1
    eigen_sum = real_eigenvals.shape[0]
    print("eigen_sum:  ", eigen_sum)
    k2_end = eigen_sum - (k2 - 1) * step - 1
    k1_list = range(0, k1_end, step)
    k2_list = range(k2_end, eigen_sum, step)
    eigenvals = torch.cat(
        [real_eigenvals[k1_list], real_eigenvals[k2_list]]
    )
    eigenvecs = torch.cat(
        [real_eigenvecs[:, k1_list], real_eigenvecs[:, k2_list]], dim=1,
    )
    print("get_syn_eigen:  Eigenvector Matrix Shape:  ", eigenvecs.shape)
    return eigenvals, eigenvecs

def get_subspace_embed(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    print("get_subspace_embed:  Eigenvector Matrix Shape:  ", eigenvecs.shape)
    print("get_subspace_embed:  x_trans Matrix Shape:  ", x_trans.shape)
    u_unsqueeze = (eigenvecs.T).unsqueeze(2) # kn1
    x_trans_unsqueeze = x_trans.unsqueeze(1) # k1d
    sub_embed = torch.bmm(u_unsqueeze, x_trans_unsqueeze)  # kn1 @ k1d = knd
    print("get_subspace_embed:  sub_embed Matrix Shape:  ", sub_embed.shape)
    return x_trans, sub_embed

def get_subspace_covariance_matrix(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = F.normalize(input=x_trans, p=2, dim=1)
    x_trans_unsqueeze = x_trans.unsqueeze(1)  # k1d
    co_matrix = torch.bmm(x_trans_unsqueeze.permute(0, 2, 1), x_trans_unsqueeze)  # kd1 @ k1d = kdd
    # print("get_subspace_covariance_matrix:  Covariance Matrix Shape:  ", co_matrix.shape)
    return co_matrix
  
def get_embed_sum(eigenvals, eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = torch.diag(1 - eigenvals) @ x_trans # kd
    embed_sum = eigenvecs @ x_trans # n1k @ kd = n1d
    # print("get_embed_sum:  Embedding Matrix Shape:  ", embed_sum.shape)
    return embed_sum

def get_embed_mean(embed_sum, label):
    class_matrix = F.one_hot(label).float()  # nc
    class_matrix = class_matrix.T  # cn
    embed_sum = class_matrix @ embed_sum  # cd
    mean_weight = (1 / class_matrix.sum(1)).unsqueeze(-1)  # c1
    embed_mean = mean_weight * embed_sum
    embed_mean = F.normalize(input=embed_mean, p=2, dim=1) # cd
    # print("get_embed_mean:  embed_mean shape:  ", embed_mean.shape)
    return embed_mean

def get_train_lcc(idx_lcc, idx_train, y_full, num_nodes, num_classes):
    idx_train_lcc = list(set(idx_train).intersection(set(idx_lcc)))
    y_full = y_full.cpu().numpy()
    if len(idx_lcc) == num_nodes:
        idx_map = idx_train
    else:
        y_train = y_full[idx_train]
        y_train_lcc = y_full[idx_train_lcc]

        y_lcc_idx = list((set(range(num_nodes)) - set(idx_train)).intersection(set(idx_lcc)))
        y_lcc_ = y_full[y_lcc_idx]
        counter_train = Counter(y_train)
        counter_train_lcc = Counter(y_train_lcc)
        idx = np.arange(len(y_lcc_))
        for c in range(num_classes):
            num_c = counter_train[c] - counter_train_lcc[c]
            if num_c > 0:
                idx_c = list(idx[y_lcc_ == c])
                idx_c = np.array(y_lcc_idx)[idx_c]
                idx_train_lcc += list(np.random.permutation(idx_c)[:num_c])
        idx_map = [idx_lcc.index(i) for i in idx_train_lcc]               
    return idx_train_lcc, idx_map

"""
Utility Functions Explanation:

1. seed_everything(seed):
   Ensures reproducibility by setting seeds for Python, NumPy, and PyTorch random number generators. It includes support for CUDA-based random seeds.

2. init_params(module):
   Initializes parameters for specific neural network layers:
   - For nn.Linear: Initializes weights using a uniform distribution and biases (if present) in the same way.
   - For nn.Embedding: Initializes weights using a normal distribution.

3. normalize_features(mx):
   Normalizes a feature matrix row-wise such that the sum of each row equals 1. This is important for ensuring comparable scales across features.

4. normalize_adj(mx):
   Normalizes an adjacency matrix with added self-loops using the formula A' = D^-1/2 * (A + I) * D^-1/2, where D is the degree matrix. This normalization is commonly used in graph-based learning to smooth over neighbors.

5. normalize_adj_to_sparse_tensor(mx):
   Combines adjacency matrix normalization (using normalize_adj) and converts it into a PyTorch SparseTensor for efficient GPU computations.

6. get_syn_eigen(real_eigenvals, real_eigenvecs, eigen_k, ratio, step=1):
   Constructs synthetic eigenvalues and eigenvectors for embedding generation by selecting a subset of eigenvalues from the spectrum based on a specified ratio. This is useful for feature dimensionality reduction or customized embeddings.

7. get_subspace_embed(eigenvecs, x):
   Projects input data `x` onto the subspace defined by the provided eigenvectors. This produces subspace embeddings that reflect the spectral properties of the graph.

8. get_subspace_covariance_matrix(eigenvecs, x):
   Computes the covariance matrix for data projected into the subspace defined by eigenvectors. This is normalized and transformed to understand feature relationships in the spectral domain.

9. get_embed_sum(eigenvals, eigenvecs, x):
   Computes an embedding by applying a transformation weighted by eigenvalues and eigenvectors. This results in embeddings that de-emphasize specific spectral components.

10. get_embed_mean(embed_sum, label):
    Computes class-wise mean embeddings by summing embeddings for nodes with the same label and normalizing the results. This is useful for analyzing class-based patterns in embedding spaces.

11. get_train_lcc(idx_lcc, idx_train, y_full, num_nodes, num_classes):
    Generates a subset of training indices within the largest connected component (LCC) of a graph:
    - Ensures that training indices are valid within the LCC.
    - Balances class representation by adding additional indices from the LCC as needed.
    - Returns the updated training indices and their mapping to the LCC.

These utilities collectively handle tasks such as preprocessing graphs, initializing models, normalizing data, and preparing embeddings. They are essential for graph-based learning workflows, ensuring efficiency, scalability, and compatibility with PyTorch.
"""

