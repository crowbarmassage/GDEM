import os
import argparse
import torch
import numpy as np
import scipy.sparse as sp

from dataset import get_dataset, get_largest_cc, get_eigh, Transd2Ind, DataGraphSAINT
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="citeseer") #[citeseer, pubmed, ogbn-arxiv, flickr, reddit, squirrel, twitch-gamer]
parser.add_argument("--normalize_features", type=bool, default=True)
args = parser.parse_args([])

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)

else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full)
print(len(data_full))
print(data_full)
#print(len(data))
dir = f"./data/{args.dataset}"
if not os.path.isdir(dir):
    os.makedirs(dir)
print(data.num_nodes)
idx_lcc, adj_norm_lcc, _ = get_largest_cc(data.adj_full, data.num_nodes, args.dataset)
np.save(f"{dir}/idx_lcc.npy", idx_lcc)
np.save(f"{dir}/adj_norm_lcc.npy", adj_norm_lcc)
# np.save("adj_norm_lcc.npy", array.astype(np.float64)) # Switch this line to fix pickle problem
# Subset the feature matrix for nodes in the LCC
if hasattr(data, 'x_full'):
    features_lcc = data.x_full[idx_lcc]
    print("Feature matrix for LCC shape:", features_lcc.shape)

    # Save the subsetted feature matrix
    np.save(f"{dir}/features_lcc.npy", features_lcc.cpu().numpy())  # Save as NumPy array
else:
    print("Feature matrix not found in the dataset.")
L_lcc = sp.eye(len(idx_lcc)) - adj_norm_lcc
print("Laplacian of largest connected component shape:  ", L_lcc.shape)    
eigenvals_lcc, eigenvecs_lcc = get_eigh(L_lcc, f"{args.dataset}", True)
print("Minimum Laplacian eigenvalue:", min(eigenvals_lcc))
print("Maximum Laplacian eigenvalue:", max(eigenvals_lcc))
np.save(f"{dir}/L_norm_lcc.npy", L_lcc)
# np.save("L_norm_lcc.npy", array.astype(np.float64)) # Switch this line to fix pickle problem

idx_train_lcc, idx_map = get_train_lcc(idx_lcc=idx_lcc, idx_train=data.idx_train, y_full=data.y_full, num_nodes=data.num_nodes, num_classes=data.num_classes)
np.save(f"{dir}/idx_train_lcc.npy", idx_train_lcc)
np.save(f"{dir}/idx_map.npy", idx_map)

# - **`idx_lcc.npy`**: This file stores the indices of the largest connected component (LCC) in the graph. The LCC refers to the largest subgraph
# in which every pair of nodes is connected either directly or via other nodes. This is essential when focusing on the primary structure of a graph
# for tasks like node classification or graph analysis while ignoring disconnected or minor subgraphs.

# - **`idx_map.npy`**: This file contains a mapping between the node indices in the original graph and the corresponding node indices in the largest
# connected component (LCC). This mapping helps translate between the indices of the full graph and the reduced LCC graph, enabling consistent node
# labeling when working within the LCC.

# - **`idx_train_lcc.npy`**: This file includes the indices of training nodes within the largest connected component.
#     These indices represent the subset of nodes in the LCC that are used for supervised tasks, such as training a model in a node classification problem.

# These files are critical in preprocessing steps where the focus is narrowed to the largest, most relevant subgraph to improve computational
# efficiency and analysis accuracy.
