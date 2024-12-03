import json
import argparse
from collections import Counter
import numpy as np
import torch

from utils import *
from dataset import get_dataset, get_largest_cc, load_eigen, Transd2Ind, DataGraphSAINT

import agent as agent
import importlib
importlib.reload(agent)
from agent import GraphAgent

# Added helper function
def get_available_gpu():
    """Get the first available GPU or return 'cpu'"""
    if torch.cuda.is_available():
        return 0
    return 'cpu'

# Add memory management function
def manage_gpu_memory():
    """Configure PyTorch memory management"""
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        # Configure memory allocator
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
        # Set memory allocation strategy
        torch.cuda.memory.set_per_process_memory_fraction(0.8)
        return True
    return False

# Process covariance matrix in chunks - do I use this?
# def process_covariance_chunks(eigenvecs, x, chunk_size):
#     n_chunks = (x.shape[0] + chunk_size - 1) // chunk_size
#     co_matrix = 0
#     for i in range(n_chunks):
#         start_idx = i * chunk_size
#         end_idx = min((i + 1) * chunk_size, x.shape[0])
#         chunk = x[start_idx:end_idx]
#         co_matrix += get_subspace_covariance_matrix(eigenvecs, chunk)
#         torch.cuda.empty_cache()  # Clear cache after each chunk
#     return co_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
#parser.add_argument("--gpu_id", type=int, default=1, help="gpu id")
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--config", type=str, default='./config/config_distill.json')

parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--dataset", type=str, default="citeseer") # [citeseer, pubmed, ogbn-arxiv, flickr, reddit, squirrel, twitch-gamer]
parser.add_argument("--normalize_features", type=bool, default=True)
parser.add_argument("--reduction_rate", type=float, default=0.1)

parser.add_argument("--evaluate_gnn", type=str, default="GCN")
parser.add_argument("--epoch_gnn", type=int, default=2000)
parser.add_argument("--nlayers", type=float, default=2)
parser.add_argument("--hidden_dim", type=int, default=256)

args = parser.parse_args([])

section=f"{args.dataset}-{str(args.reduction_rate)}"

with open(args.config, "r") as config_file:
    config = json.load(config_file)
if section in config:
    config = config[section]
for key, value in config.items():
    setattr(args, key, value)

print(args)

manage_gpu_memory() # New line

#torch.cuda.set_device(args.gpu_id)
if torch.cuda.is_available():
    device = torch.device(f'cuda:{get_available_gpu()}')
    print(f"Using GPU device {get_available_gpu()}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
else:
    device = torch.device('cpu')
    print("Using CPU")
seed_everything(args.seed)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)

else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full)

dataset_dir = f"./data/{args.dataset}"
idx_lcc = np.load(f"{dataset_dir}/idx_lcc.npy")
idx_train_lcc = np.load(f"{dataset_dir}/idx_train_lcc.npy")
idx_map = np.load(f"{dataset_dir}/idx_map.npy")

eigenvals_lcc, eigenvecs_lcc = load_eigen(args.dataset)
eigenvals_lcc = torch.FloatTensor(eigenvals_lcc)
eigenvecs_lcc = torch.FloatTensor(eigenvecs_lcc)

n_syn = int(len(data.idx_train) * args.reduction_rate)
args.eigen_k = args.eigen_k if args.eigen_k < n_syn else n_syn

# Add these print statements to distill.py
print(f"Training set size: {len(data.idx_train)}")
print(f"Reduction rate: {args.reduction_rate}")
print(f"Number of synthetic nodes (n_syn): {n_syn}")

# Also check eigenvalues size:
print(f"eigen_k: {args.eigen_k}")

CHUNK_SIZE=1000  # New line
eigenvals, eigenvecs = get_syn_eigen(real_eigenvals=eigenvals_lcc, real_eigenvecs=eigenvecs_lcc, eigen_k=args.eigen_k, ratio=args.ratio)
print("Eigenvecs Shape:  ", eigenvecs.shape)
print("Full x Shape:  ", data.x_full[idx_lcc].shape)
co_x_trans_real = get_subspace_covariance_matrix(eigenvecs, data.x_full[idx_lcc]) #kdd
embed_sum = get_embed_sum(eigenvals=eigenvals, eigenvecs=eigenvecs, x=data.x_full[idx_lcc])
embed_sum = embed_sum[idx_map,:]
embed_mean_real = get_embed_mean(embed_sum=embed_sum, label=data.y_full[idx_train_lcc])

# data = data.to(device)
# eigenvals = eigenvals.to(device)
# co_x_trans_real = co_x_trans_real.to(device)
# embed_mean_real = embed_mean_real.to(device)
data = data.cuda()
eigenvals = eigenvals.cuda()
co_x_trans_real = co_x_trans_real.cuda()
embed_mean_real = embed_mean_real.cuda()

# Add debug information
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
accs = []
for ep in range(args.runs):
    torch.cuda.empty_cache()  # Clear cache before each run
    args.expID = ep
    agent = GraphAgent(args, data)
    try:
        acc = agent.train(eigenvals, co_x_trans_real, embed_mean_real)
        accs.append(acc)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Out of memory in run {ep}. Clearing cache and continuing...")
            torch.cuda.empty_cache()
            continue
        else:
            raise e
    # acc = agent.train(eigenvals, co_x_trans_real, embed_mean_real)
    # accs.append(acc)

mean_acc = np.mean(accs)
std_acc = np.std(accs)
print(f"Mean ACC: {mean_acc}\t Std: {std_acc}")
