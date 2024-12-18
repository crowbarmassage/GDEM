import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import os
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from model.gcn import *
import agent as agent
import importlib
importlib.reload(agent)
from agent import GraphAgent

# Data Loading Functions

def load_constant_files(file_path, device):
    """
    Load files that don't depend on parameters x and y
    
    Args:
        file_path (str): Path to the directory containing the files
        device (torch.device): Device to load the tensors to
        
    Returns:
        tuple: Tuple containing (A, idx_train, idx_features, L_eigenvectors, L_eigenvalues)
    """
    # Load constant files
    A = np.load(file_path + 'adj_norm_lcc.npy', allow_pickle=True)
    idx_train = np.load(file_path + 'idx_lcc.npy', allow_pickle=True)
    idx_features = np.load(file_path + 'features_lcc.npy', allow_pickle=True)
    A = sp.csr_matrix(A.item())
    L_eigenvectors = np.load(file_path + 'eigenvectors.npy', allow_pickle=True)
    L_eigenvalues = np.load(file_path + 'eigenvalues.npy', allow_pickle=True)
    
    # Convert sparse matrices to dense if needed
    if sp.issparse(A):
        A = A.toarray()
        print("Converted A to dense matrix")
    if sp.issparse(L_eigenvectors):
        L_eigenvectors = L_eigenvectors.toarray()
        print("Converted L_eigenvectors to dense matrix")
    if sp.issparse(L_eigenvalues):
        L_eigenvalues = L_eigenvalues.toarray()
        print("Converted L_eigenvalues to dense matrix")
        
    return A, idx_train, idx_features, L_eigenvectors, L_eigenvalues

def load_parameter_dependent_files(file_path, device, x, y):
    """
    Load files that depend on parameters x and y
    
    Args:
        file_path (str): Path to the directory containing the files
        device (torch.device): Device to load the tensors to
        x (int): Parameter for selecting specific files
        y (float): Parameter for x_init file
        
    Returns:
        tuple: Tuple containing (eigenvals, eigenvecs, feat, x_init, A_distilled)
    """
    # Load parameter-dependent files
    eigenvals = torch.load(f'{file_path}eigenvals_syn_{x}.pt', map_location=device, weights_only=True)
    eigenvecs = torch.load(f'{file_path}eigenvecs_syn_{x}.pt', map_location=device, weights_only=True)
    feat = torch.load(f'{file_path}feat_{x}.pt', map_location=device, weights_only=True)
    x_init = torch.load(f'{file_path}x_init_{y}_{x}.pt', map_location=device, weights_only=True)
    
    # Print shapes for verification
    print(f"Eigenvalues of Distilled Laplacian: {eigenvals.shape}")
    print(f"Eigenvectors of Distilled: {eigenvecs.shape}")
    print(f"Final X of Distilled: {feat.shape}")
    print(f"Initial X of Distilled: {x_init.shape}")
    
    # Create diagonal matrix and compute A_distilled
    diagonal_matrix = torch.diag(1 - eigenvals)
    A_distilled = torch.mm(torch.mm(eigenvecs, diagonal_matrix), eigenvecs.T)
    
    print(f"Diagonal Matrix (1 - eigenvalues): {diagonal_matrix.shape}")
    print(f"Normalized Adjacency Matrix (A_distilled): {A_distilled.shape}")
    
    return eigenvals, eigenvecs, feat, x_init, A_distilled

def compute_reconstruction_residual(L_eigenvectors, A_distilled, A, k, device=None, dtype=torch.float32):
    """
    Compute the reconstruction and residual using the first k eigenvectors
    
    Args:
        L_eigenvectors (numpy.ndarray): Matrix of eigenvectors
        A_distilled (torch.Tensor): Distilled adjacency matrix
        A (numpy.ndarray): Original adjacency matrix
        k (int): Number of eigenvectors to use
        device (str): Device to use. If None, will use CUDA if available, else CPU
        dtype (torch.dtype): Dtype to use for tensors (default: torch.float32)
    """
    # Handle device selection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Get first k eigenvectors and move to device with specified dtype
    V = L_eigenvectors[:, :k]  # First k eigenvectors
    V = torch.tensor(V, dtype=dtype).to(device)
    print(f"V shape: {V.shape}, dtype: {V.dtype}")
    
    # Ensure A_distilled is on the same device and dtype
    A_distilled = A_distilled.to(device).to(dtype)
    print(f"A_distilled shape: {A_distilled.shape}, dtype: {A_distilled.dtype}")
    
    # Convert sparse matrix to dense if needed
    if sp.issparse(A):
        A = A.toarray()
        print("Converted A to dense matrix")
    
    # Ensure A is numeric and convert to tensor with specified dtype
    A = np.array(A, dtype=np.float32 if dtype == torch.float32 else np.float64)
    A_torch = torch.tensor(A, dtype=dtype).to(device)
    
    # Compute reconstruction
    A_reconstructed = V @ A_distilled @ V.T
    
    # Compute residual
    R = A_torch - A_reconstructed
    print(f"Residual shape: {R.shape}, dtype: {R.dtype}")
    
    return V, A_torch, A_reconstructed, R

def analyze_residual_matrix(R, num_bins=50, threshold=0.01, return_hist_data=True):
    """
    Analyze residual matrix by creating a histogram and sparsifying based on threshold
    
    Args:
        R (torch.Tensor): Residual matrix to analyze
        num_bins (int): Number of bins for histogram analysis
        threshold (float): Threshold for sparsification
        return_hist_data (bool): Whether to return histogram data
        
    Returns:
        tuple: (R_sparsified, results_dict) where:
            - R_sparsified: Sparsified version of residual matrix
            - results_dict: Dictionary containing analysis results including:
                - nonzero_count: Number of nonzero elements after sparsification
                - zero_count: Number of zero elements after sparsification
                - hist_counts: Counts for each histogram bin (if return_hist_data=True)
                - bin_edges: Edges of histogram bins (if return_hist_data=True)
                - sparsification_ratio: Percentage of elements set to zero
    """
    # Step 1: Flatten the matrix for histogram analysis
    R_flattened = R.flatten()
    
    # Step 2: Create histogram
    hist_counts, bin_edges = np.histogram(R_flattened.cpu().numpy(), bins=num_bins)
    
    # Step 3: Print bin information
    for i in range(len(hist_counts)):
        print(f"Bin {i + 1}: [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}] - Count: {hist_counts[i]}")
    
    # Step 4: Sparsify the residual matrix
    R_sparsified = R.clone()  # Create a copy to avoid modifying the original
    R_sparsified[(R_sparsified > -threshold) & (R_sparsified < threshold)] = 0
    
    # Calculate statistics
    num_nonzero = torch.count_nonzero(R_sparsified).item()
    num_zero = R_sparsified.numel() - num_nonzero
    sparsification_ratio = (num_zero / R_sparsified.numel()) * 100
    
    # Print results
    print(f"\nSparsification Results (threshold = {threshold}):")
    print(f"Number of nonzero elements: {num_nonzero}")
    print(f"Number of zero elements: {num_zero}")
    print(f"Sparsification ratio: {sparsification_ratio:.2f}%")
    
    # Prepare return dictionary
    results = {
        'nonzero_count': num_nonzero,
        'zero_count': num_zero,
        'sparsification_ratio': sparsification_ratio
    }
    
    if return_hist_data:
        results.update({
            'hist_counts': hist_counts,
            'bin_edges': bin_edges
        })
    frobenius_norm = torch.norm(R_sparsified, p='fro')
    return R_sparsified, frobenius_norm, results

def plot_residual_histogram(R_flattened, num_bins=50, figsize=(10, 6)):
    """
    Plot histogram of residual matrix elements
    
    Args:
        R_flattened (torch.Tensor): Flattened residual matrix
        num_bins (int): Number of bins for histogram
        figsize (tuple): Figure size for the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    plt.hist(R_flattened.cpu().numpy(), bins=num_bins, color="blue", alpha=0.7)
    plt.title("Histogram of Matrix Elements")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def compute_new_connections(cluster_connections, x_syn, features, method='combined'):
    """Compute connections between new and existing synthetic nodes"""
    # Get device and dtype from x_syn
    device = x_syn.device
    dtype = x_syn.dtype
    
    # Convert cluster_connections to tensor on the correct device right at the start
    cluster_connections = torch.tensor(cluster_connections, dtype=dtype, device=device) if not torch.is_tensor(cluster_connections) else cluster_connections.to(device)
    
    # Also ensure x_syn is on the right device
    x_syn = x_syn.to(device)
    
    if method == 'residual':
        syn_patterns = project_synthetic_to_original(x_syn, features)  # Already device-aware
        # Move data to CPU only for numpy operation
        cluster_connections_cpu = cluster_connections.cpu().numpy()
        syn_patterns_cpu = syn_patterns.cpu().numpy()
        connections = torch.tensor(
            np.dot(cluster_connections_cpu, syn_patterns_cpu.T), 
            dtype=dtype, 
            device=device
        )
    elif method == 'feature':
        features = torch.tensor(features, dtype=dtype, device=device) if not torch.is_tensor(features) else features.to(device)
        connections = 1 - torch.cdist(cluster_connections @ features, x_syn)
    else:  # combined
        syn_patterns = project_synthetic_to_original(x_syn, features)
        
        # CPU operations
        cluster_connections_cpu = cluster_connections.cpu().numpy()
        syn_patterns_cpu = syn_patterns.cpu().numpy()
        residual_sim = torch.tensor(
            np.dot(cluster_connections_cpu, syn_patterns_cpu.T),
            dtype=dtype,
            device=device
        )
        
        # GPU operations
        features = torch.tensor(features, dtype=dtype, device=device) if not torch.is_tensor(features) else features.to(device)
        feature_sim = 1 - torch.cdist(cluster_connections @ features, x_syn)
        
        connections = 0.5 * (residual_sim + feature_sim)
    
    connections = torch.sigmoid(connections)
    connections = threshold_sparse(connections, sparsity=0.7)
    
    return connections

def compute_new_to_new_connections(cluster_connections, k):
    """Compute connections between new nodes"""
    # Get device from existing tensor or use same as cluster_connections
    if torch.is_tensor(cluster_connections):
        device = cluster_connections.device
        dtype = cluster_connections.dtype
        cluster_connections = cluster_connections.to(device)
    else:
        # If not a tensor, use CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32
        cluster_connections = torch.tensor(cluster_connections, dtype=dtype, device=device)
    
    # Compute similarity
    pattern_sim = torch.mm(cluster_connections, cluster_connections.t())
    
    # Normalize by pattern magnitudes
    norms = torch.norm(cluster_connections, dim=1)
    pattern_sim = pattern_sim / (torch.outer(norms, norms) + 1e-8)
    
    # Convert to tensor and normalize
    connections = torch.sigmoid(pattern_sim)
    
    # Make symmetric
    connections = 0.5 * (connections + connections.t())
    
    # Set diagonal to 0
    connections.fill_diagonal_(0)
    
    # Sparsify while ensuring connectivity
    #connections = ensure_sparse_connectivity(connections)
    connections = ensure_connectivity(connections)
   
    return connections

def project_synthetic_to_original(x_syn, features):
    """Project synthetic nodes into original space using feature similarity"""
    # Get device and dtype from x_syn
    device = x_syn.device
    dtype = x_syn.dtype
    
    # Convert features to tensor with explicit device placement
    if not torch.is_tensor(features):
        features_tensor = torch.tensor(features, dtype=dtype, device=device)
    else:
        features_tensor = features.to(device)
    
    sim_matrix = torch.cdist(x_syn, features_tensor)
    weights = torch.softmax(-sim_matrix, dim=1)
    return weights

def threshold_sparse(matrix, sparsity=0.7):
    """Threshold matrix to achieve desired sparsity"""
    values = matrix.reshape(-1)
    threshold = torch.quantile(values, sparsity)
    return matrix * (matrix > threshold)

def ensure_sparse_connectivity(adj_matrix):
    """Ensure graph is connected while maintaining sparsity"""
    # Get the device from the input matrix
    device = adj_matrix.device
    
    # Move to CPU for numpy operations
    edge_weights = -adj_matrix.cpu().numpy()
    mst = minimum_spanning_tree(edge_weights).toarray()
    
    # Create tensor from MST on the correct device
    mst_tensor = torch.tensor(mst + mst.T > 0, device=device)
    
    # Threshold on the same device
    sparse_adj = threshold_sparse(adj_matrix)
    
    # Combine using tensors on the same device
    connected_adj = ((sparse_adj > 0) | mst_tensor).float()
    
    return connected_adj

def ensure_connectivity(adj_matrix):
    """Ensure graph is connected without enforcing sparsity"""
    # Get the device from the input matrix
    device = adj_matrix.device
    
    # Get original edge information
    num_nodes = adj_matrix.shape[0]
    num_edges = torch.sum(adj_matrix > 0).item()
    print(f"\nOriginal Graph:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    
    # Move to CPU for numpy operations
    edge_weights = -adj_matrix.cpu().numpy()
    mst = minimum_spanning_tree(edge_weights).toarray()
    
    # Create tensor from MST on the correct device
    mst_tensor = torch.tensor(mst + mst.T > 0, device=device)
    
    # Combine original adjacency with MST to ensure connectivity
    connected_adj = ((adj_matrix > 0) | mst_tensor).float()
    
    # Get final edge information
    final_num_edges = torch.sum(connected_adj > 0).item()
    print(f"\nConnected Graph:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {final_num_edges}")
    print(f"Added edges: {final_num_edges - num_edges}")
    
    return connected_adj

def extend_spectral_properties(orig_eigenvals, orig_eigenvecs, A_aug, features, n_orig=2120, n_syn_old=12, n_syn_new=0):
    """
    Extend spectral properties to include new synthetic nodes while maintaining full dimensions
    
    Parameters:
    - orig_eigenvals: Original eigenvalues (n_syn_old,)
    - orig_eigenvecs: Original eigenvectors (n_orig x n_syn_old)
    - A_aug: Augmented adjacency matrix ((n_syn_old + n_syn_new) x (n_syn_old + n_syn_new))
    - features: Original node features (n_orig x d)
    - n_orig: Number of original nodes (2120)
    - n_syn_old: Number of original synthetic nodes (12)
    - n_syn_new: Number of new synthetic nodes (2)
    
    Returns:
    - aug_eigenvals: Augmented eigenvalues (n_syn_old + n_syn_new,)
    - aug_eigenvecs: Augmented eigenvectors (n_orig x (n_syn_old + n_syn_new))
    """
    # Initialize augmented eigenvectors
    aug_eigenvecs = torch.zeros((n_orig, n_syn_old + n_syn_new))
    
    # Copy original eigenvectors
    aug_eigenvecs[:, :n_syn_old] = orig_eigenvecs
    
    # For each new synthetic node
    for i in range(n_syn_new):
        # Project original nodes onto new synthetic node space
        projection = compute_node_projection(features, A_aug[n_syn_old + i, :n_syn_old])
        
        # Add new eigenvector column
        aug_eigenvecs[:, n_syn_old + i] = projection
    
    # Orthogonalize the new vectors with respect to existing ones
    aug_eigenvecs = gram_schmidt(aug_eigenvecs)
    
    # Compute new eigenvalues using Rayleigh quotient
    aug_eigenvals = compute_new_eigenvalues(aug_eigenvecs, A_aug, orig_eigenvals)
    
    return aug_eigenvals, aug_eigenvecs

def compute_node_projection(features, connections):
    """
    Compute projection of original nodes onto new synthetic node
    
    Parameters:
    - features: Original node features (n_orig x d)
    - connections: Connection strengths to existing synthetic nodes
    
    Returns:
    - projection: New eigenvector column (n_orig,)
    """
    # Use feature similarity and connection patterns
    feature_sim = torch.mm(features, features.t())
    
    # Combine with connection information
    projection = feature_sim @ connections
    
    # Normalize
    projection = projection / torch.norm(projection)
    
    return projection

def gram_schmidt(vectors):
    """
    Orthogonalize vectors while maintaining dimensions
    """
    n, k = vectors.shape
    orthogonal = torch.zeros_like(vectors)
    
    for i in range(k):
        v = vectors[:, i].clone()
        # Subtract projections onto previous vectors
        for j in range(i):
            v = v - (v @ orthogonal[:, j]) * orthogonal[:, j]
        # Normalize
        if torch.norm(v) > 1e-10:
            v = v / torch.norm(v)
        orthogonal[:, i] = v
    return orthogonal

def compute_new_eigenvalues(eigenvecs, A_aug, trunc_eigenvals):
    """
    Compute eigenvalues for augmented system using Rayleigh quotient
    """
    n_total = A_aug.shape[0]
    aug_eigenvals = torch.zeros(n_total)
    
    # Keep original eigenvalues
    aug_eigenvals[:len(trunc_eigenvals)] = trunc_eigenvals
    
    # Compute new eigenvalues
    for i in range(len(trunc_eigenvals), n_total):
        v = eigenvecs[:, i]
        # Rayleigh quotient
        aug_eigenvals[i] = (v @ A_aug @ v) / (v @ v)
    
    return aug_eigenvals

def augment_graph_louvain(residual, features, x_syn, A_distilled, num_new_nodes=18, orig_eigenvals=None, orig_eigenvecs=None):
    """
    Modified to select num_new_nodes clusters with highest Frobenius norm contribution
    """
    # Convert tensors to numpy
    device = x_syn.device
    if torch.is_tensor(residual):
        residual_np = residual.cpu().numpy()
    else:
        residual_np = residual
        
    if torch.is_tensor(features):
        features_np = features.cpu().numpy()
    else:
        features_np = features
    
    # Create networkx graph from residual
    similarity = np.abs(residual_np)
    G = nx.from_numpy_array(similarity)
    
    # Apply Louvain clustering
    communities = nx.community.louvain_communities(G)
    
    # Convert communities to labels
    n_nodes = len(G)
    cluster_labels = np.zeros(n_nodes, dtype=int)
    for i, community in enumerate(communities):
        for node in community:
            cluster_labels[node] = i
    
    # Calculate Frobenius norm contribution for each cluster
    unique_labels = np.unique(cluster_labels)
    cluster_contributions = []
    
    print(f"\nTotal communities found: {len(communities)}")
    total_frob = np.linalg.norm(residual_np, 'fro')**2
    
    for label in unique_labels:
        cluster_mask = cluster_labels == label
        cluster_size = np.sum(cluster_mask)
        
        # Calculate cluster's contribution
        cluster_residual = residual_np[cluster_mask]
        cluster_frob = np.linalg.norm(cluster_residual, 'fro')**2
        contribution_percent = (cluster_frob / total_frob) * 100
        
        cluster_contributions.append({
            'label': label,
            'size': cluster_size,
            'contribution': contribution_percent,
            'features': features_np[cluster_mask].mean(axis=0),
            'connectivity': residual_np[cluster_mask].mean(axis=0)
        })
        
        print(f"\nCluster {label}:")
        print(f"Size: {cluster_size} nodes ({(cluster_size/n_nodes)*100:.1f}% of graph)")
        print(f"Contribution to Frobenius norm: {contribution_percent:.2f}%")
    
    # Sort clusters by contribution and take top num_new_nodes
    cluster_contributions.sort(key=lambda x: x['contribution'], reverse=True)
    selected_clusters = cluster_contributions[:num_new_nodes]
    
    k = len(selected_clusters)
    if k == 0:
        print("\nNo clusters to process")
        # if orig_eigenvals is not None and orig_eigenvecs is not None:
        #     return cluster_labels, A_distilled, x_syn, orig_eigenvals, orig_eigenvecs
        return cluster_labels, A_distilled, x_syn
    
    print(f"\nCreating {k} new synthetic nodes from top contributing clusters:")
    for cluster in selected_clusters:
        print(f"Cluster {cluster['label']}: {cluster['contribution']:.2f}% contribution")
    
    # Create augmented features
    new_features = torch.tensor(np.stack([c['features'] for c in selected_clusters])).float().to(device)  # Move to same device as x_syn
    x_aug = torch.vstack([x_syn, new_features])
    
    # Create augmented adjacency matrix
    n_syn = len(x_syn)
    A_aug = torch.zeros((n_syn + k, n_syn + k), device=device)  # Create on same device
    
    # Keep existing synthetic node connections
    A_aug[:n_syn, :n_syn] = A_distilled
    
    # Add connections for new nodes
    cluster_connections = np.stack([c['connectivity'] for c in selected_clusters])
    
    new_to_existing = compute_new_connections(cluster_connections, x_syn, features).to(device)  # Move to same device
    A_aug[n_syn:, :n_syn] = new_to_existing
    A_aug[:n_syn, n_syn:] = new_to_existing.T
    
    new_to_new = compute_new_to_new_connections(cluster_connections, k).to(device)  # Move to same device
    A_aug[n_syn:, n_syn:] = new_to_new
    
    return cluster_labels, A_aug, x_aug
    
def compute_augmented_reconstruction(V, aug_A, A_orig, k, threshold=0.01):
   """
   Compute augmented distilled matrix reconstruction and analysis
   
   Parameters:
   - V: nxk eigenvector matrix for reconstruction
   - aug_A: (k+m)x(k+m) augmented distilled matrix
   - A_orig: nxn original adjacency matrix
   - k: number of nodes in augmented matrix (k+m)
   - threshold: sparsification threshold
   
   Returns:
   - aug_A: (k+m)x(k+m) augmented matrix
   - R_sparsified: nxn sparsified residual
   - results: dictionary of metrics
   """
   # Get device and dtype from input tensors
   device = A_orig.device
   dtype = A_orig.dtype
   
   # Ensure all tensors are on same device and dtype
   V = V.to(device=device, dtype=dtype)
   aug_A = aug_A.to(device=device, dtype=dtype)
   
   # Compute reconstruction: V @ aug_A @ V.T
   A_reconstructed = torch.mm(torch.mm(V, aug_A), V.T)
   
   # Compute residual
   R = A_orig - A_reconstructed
   
   # Sparsify residual
   R_sparsified = R.clone()
   R_sparsified[(R_sparsified > -threshold) & (R_sparsified < threshold)] = 0
   
   # Compute metrics
   metrics = {
       'nonzero_count': torch.count_nonzero(R_sparsified).item(),
       'zero_count': R_sparsified.numel() - torch.count_nonzero(R_sparsified).item(),
       'frobenius_norm': torch.norm(R_sparsified, p='fro').item(),
       'reconstruction': A_reconstructed,
       'residual': R
   }
   
   metrics['sparsification_ratio'] = (metrics['zero_count'] / R_sparsified.numel()) * 100
   
   # Print summary
   print(f"\nReconstruction Results:")
   print(f"Augmented matrix size: {k}x{k}")
   print(f"Nonzero elements in residual: {metrics['nonzero_count']}")
   print(f"Frobenius norm of residual: {metrics['frobenius_norm']:.4f}")
   print(f"Sparsification ratio: {metrics['sparsification_ratio']:.2f}%")
   
   return aug_A, R_sparsified, metrics

def reassign_augmented_graph_labels(x_aug, A_aug, eigenvecs_aug, x_train, y_train, reduction_rate, alpha=0.5):
   """
   Args:
       x_aug (torch.Tensor): Features of augmented nodes [n_aug x feat_dim]
       A_aug (torch.Tensor): Augmented adjacency matrix [n_aug x n_aug]
       eigenvecs_aug (torch.Tensor): Augmented eigenvectors [n_orig x n_aug]
       x_train (torch.Tensor): Training node features [n_train x feat_dim]
       y_train (torch.Tensor): Training node labels [n_train]
       reduction_rate (float): Target size relative to original graph
       alpha (float): Weight between feature and structural similarity (0-1)
   """
   # Get device and move tensors
   device = x_aug.device
   x_aug = x_aug.to(device)
   A_aug = A_aug.to(device)
   eigenvecs_aug = eigenvecs_aug.to(device)
   x_train = x_train.to(device)
   y_train = y_train.to(device)

   n_aug = len(x_aug)      # Number of augmented nodes
   n_train = len(x_train)  # Number of training nodes
   
   print(f"x_aug shape: {x_aug.shape}")
   print(f"x_train shape: {x_train.shape}")
   print(f"eigenvecs_aug shape: {eigenvecs_aug.shape}")

   # Calculate target distribution
   class_counts = Counter(y_train.cpu().numpy())
   sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
   
   # Calculate target counts for each class
   target_counts = {}
   running_sum = 0
   for i, (c, count) in enumerate(sorted_classes):
       if i == len(sorted_classes) - 1:
           target_counts[c] = n_aug - running_sum
       else:
           min_nodes = max(int(count * reduction_rate), 1)
           aug_scale = n_aug / int(len(y_train) * reduction_rate)
           target_counts[c] = max(int(min_nodes * aug_scale), 1)
           running_sum += target_counts[c]
   
   # Feature similarity [n_aug x n_train]
   feat_sim = F.normalize(torch.mm(x_aug, x_train.t()), dim=1)
   print(f"feat_sim shape: {feat_sim.shape}")

   # Structural similarity
   # 1. Get structural embedding for augmented nodes using A_aug
   aug_struct = F.normalize(A_aug, dim=1)  # [n_aug x n_aug]
   
   # 2. Get structural embedding for training nodes using eigenvectors
   train_indices = torch.arange(n_train)  # Indices of training nodes
   train_struct = eigenvecs_aug[train_indices, :A_aug.shape[1]]  # [n_train x n_aug]
   train_struct = F.normalize(train_struct, dim=1)
   
   # 3. Compute structural similarity
   struct_sim = torch.mm(aug_struct, train_struct.t())  # [n_aug x n_train]
   print(f"struct_sim shape: {struct_sim.shape}")

   # Combine similarities
   combined_sim = alpha * feat_sim + (1 - alpha) * struct_sim

   # Initialize results tensors
   assigned_labels = torch.zeros(n_aug, dtype=torch.long, device=device)
   assigned_mask = torch.zeros(n_aug, dtype=torch.bool, device=device)
   current_counts = Counter()

   # First pass: Assign highest confidence predictions while respecting class balance
   confidence_scores, predicted_labels = torch.max(combined_sim, dim=1)
   sorted_indices = torch.argsort(confidence_scores, descending=True)
   
   print("\nAssigning labels in order of confidence...")
   for idx in sorted_indices:
       if assigned_mask[idx]:
           continue
       pred_label = y_train[predicted_labels[idx]].item()
       if current_counts[pred_label] < target_counts[pred_label]:
           assigned_labels[idx] = pred_label
           assigned_mask[idx] = True
           current_counts[pred_label] += 1
   
   # Second pass: Assign remaining nodes to maintain class balance
   remaining_indices = (~assigned_mask).nonzero().squeeze()
   if len(remaining_indices.shape) == 0:
       remaining_indices = remaining_indices.unsqueeze(0)
       
   print("\nBalancing remaining assignments...")
   for c, target in target_counts.items():
       if current_counts[c] >= target:
           continue
           
       # Find nodes most similar to class c
       class_mask = (y_train == c)
       class_sim = combined_sim[:, class_mask].mean(dim=1)
       
       # Sort remaining nodes by similarity to class c
       remaining_sim = class_sim[remaining_indices]
       sorted_remaining = remaining_indices[torch.argsort(remaining_sim, descending=True)]
       
       # Assign needed number of nodes
       needed = target - current_counts[c]
       for idx in sorted_remaining[:needed]:
           assigned_labels[idx] = c
           assigned_mask[idx] = True
           current_counts[c] += 1
           remaining_indices = remaining_indices[remaining_indices != idx]

   # Print distribution statistics
   print("\nFinal Label Distribution:")
   print(f"{'Class':<8} {'Target':<8} {'Assigned':<8} {'Diff':<8}")
   print("-" * 32)
   for c in sorted(target_counts.keys()):
       diff = current_counts[c] - target_counts[c]
       print(f"{c:<8} {target_counts[c]:<8} {current_counts[c]:<8} {diff:<8}")
   
   confidence_by_class = {}
   for c in target_counts.keys():
       mask = assigned_labels == c
       if torch.any(mask):
           conf = confidence_scores[mask].mean().item()
           confidence_by_class[c] = conf
   
   print("\nAverage Assignment Confidence by Class:")
   for c, conf in confidence_by_class.items():
       print(f"Class {c}: {conf:.4f}")

   return assigned_labels
    
def evaluate_augmented_graph(data, args, aug_features, aug_A, aug_labels, device='cuda'):
    """
    Evaluate augmented distilled graph on test set with flexible device selection
    
    Args:
        data: Dataset object containing test data
        args: Arguments containing model parameters
        aug_features (torch.Tensor): Features of augmented graph
        aug_A (torch.Tensor): Adjacency matrix of augmented graph
        aug_labels (torch.Tensor): Labels of augmented graph nodes
        device (str): Device to run evaluation on ('cuda' or 'cpu')
    """
    # Check if CUDA is available when device='cuda' is requested
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        device = 'cpu'
    
    # Move everything to specified device
    aug_features = aug_features.to(device)
    aug_A = aug_A.to(device)
    aug_labels = aug_labels.to(device)
    
    # Initialize model based on args.evaluate_gnn
    if args.evaluate_gnn == "GCN":
        model = GCN(
            num_features=aug_features.shape[1],
            num_classes=data.num_classes,
            hidden_dim=args.hidden_dim,
            nlayers=args.nlayers,
            dropout=args.dropout,
            lr=args.lr_gnn,
            weight_decay=args.wd_gnn
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.evaluate_gnn}")
    
    # Train model on augmented graph
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_gnn, weight_decay=args.wd_gnn)
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    
    print(f"\nTraining on {device}")
    
    for epoch in range(args.epoch_gnn):
        # Train on augmented graph
        model.train()
        optimizer.zero_grad()
        out = model(aug_features, aug_A)
        loss = F.nll_loss(out, aug_labels)
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation and test sets
        model.eval()
        with torch.no_grad():
            # Move full data to device
            x_full = data.x_full.to(device)
            adj_full = normalize_adj_to_sparse_tensor(data.adj_full).to(device)
            y_full = data.y_full.to(device)
            
            # Get predictions on full graph
            out = model(x_full, adj_full)
            
            # Move predictions to CPU for accuracy calculation
            val_preds = out[data.idx_val].max(1)[1].cpu()
            test_preds = out[data.idx_test].max(1)[1].cpu()
            
            # Compute accuracies
            val_acc = accuracy_score(y_full[data.idx_val].cpu(), val_preds)
            test_acc = accuracy_score(y_full[data.idx_test].cpu(), test_preds)
            
            # Track best accuracies
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch:4d}, Loss: {loss.item():.4f}, '
                  f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    print(f'\nBest results:')
    print(f'Val Acc: {best_val_acc:.4f}')
    print(f'Test Acc: {best_test_acc:.4f}')
    
    return best_val_acc, best_test_acc
