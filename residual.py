import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import os
from collections import Counter

torch.set_printoptions(sci_mode=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Using CPU")

# Update this path based on your file location
file_path = '/content/drive/MyDrive/All_DataAnalytics_AI/CS_514/Project/saved_citeseer/citeseer-0.1/'

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

def compute_reconstruction_residual(L_eigenvectors, A_distilled, A, k):
    """
    Compute the reconstruction and residual using the first k eigenvectors
    
    Args:
        L_eigenvectors (numpy.ndarray): Matrix of eigenvectors
        A_distilled (torch.Tensor): Distilled adjacency matrix
        A (numpy.ndarray): Original adjacency matrix
        k (int): Number of eigenvectors to use (nodes in distilled graph)
        
    Returns:
        tuple: Tuple containing (V, A_reconstructed, R)
            - V: First k eigenvectors as torch tensor
            - A_reconstructed: Reconstructed adjacency matrix
            - R: Residual matrix
    """
    # Get first k eigenvectors
    V = L_eigenvectors[:, :k]  # First k eigenvectors
    V = torch.tensor(V, dtype=torch.float64)
    print(f"V shape: {V.shape}")
    print(f"A_distilled shape: {A_distilled.shape}")
    
    # Ensure correct dtype
    A_distilled = A_distilled.to(dtype=torch.float64)
    
    # Convert sparse matrix to dense if needed
    if sp.issparse(A):
        A = A.toarray()
        print("Converted A to dense matrix")
    
    # Ensure A is numeric
    A = np.array(A, dtype=np.float64)
    
    # Compute reconstruction
    A_reconstructed = V @ A_distilled @ V.T
    
    # Convert A to PyTorch tensor and compute residual
    A_torch = torch.tensor(A, dtype=torch.float64)
    R = A_torch - A_reconstructed
    print(f"Residual shape: {R.shape}")
    
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
    """
    Compute connections between new and existing synthetic nodes (k x 12)
    
    Parameters:
    - cluster_connections: Average residual patterns for each cluster (k x 2120)
    - x_syn: Existing synthetic features (12 x d)
    - features: Original features (2120 x d)
    - method: How to compute connections ('residual', 'feature', or 'combined')
    
    Returns:
    - Connection matrix between new and existing nodes (k x 12)
    """
    k = len(cluster_connections)
    n_syn = len(x_syn)
    
    if method == 'residual':
        # Project synthetic nodes into original space based on feature similarity
        syn_patterns = project_synthetic_to_original(x_syn, features)  # (12 x 2120)
        
        # Compute similarity between cluster patterns and synthetic patterns
        connections = torch.tensor(
            np.dot(cluster_connections, syn_patterns.T)  # (k x 12)
        ).float()
        
    elif method == 'feature':
        # Direct feature similarity
        connections = 1 - torch.cdist(
            torch.tensor(cluster_connections @ features).float(), 
            x_syn
        )  # (k x 12)
        
    else:  # combined
        # Combine both residual and feature similarities
        syn_patterns = project_synthetic_to_original(x_syn, features)
        residual_sim = torch.tensor(
            np.dot(cluster_connections, syn_patterns.T)
        ).float()
        
        feature_sim = 1 - torch.cdist(
            torch.tensor(cluster_connections @ features).float(), 
            x_syn
        )
        
        connections = 0.5 * (residual_sim + feature_sim)
    
    # Normalize and threshold
    connections = torch.sigmoid(connections)  # Scale to [0,1]
    connections = threshold_sparse(connections, sparsity=0.7)
    
    return connections

def compute_new_to_new_connections(cluster_connections, k):
    """
    Compute connections between new nodes (k x k)
    
    Parameters:
    - cluster_connections: Average residual patterns for each cluster (k x 2120)
    - k: Number of new nodes
    
    Returns:
    - Adjacency matrix for new nodes (k x k)
    """
    # Compute similarity between cluster residual patterns
    pattern_sim = np.dot(cluster_connections, cluster_connections.T)  # (k x k)
    
    # Normalize by pattern magnitudes (cosine similarity)
    norms = np.linalg.norm(cluster_connections, axis=1)
    pattern_sim = pattern_sim / (norms[:, None] * norms[None, :])
    
    # Convert to tensor and normalize
    connections = torch.tensor(pattern_sim).float()
    connections = torch.sigmoid(connections)
    
    # Make symmetric
    connections = 0.5 * (connections + connections.T)
    
    # Set diagonal to 0 (no self-loops)
    connections.fill_diagonal_(0)
    
    # Sparsify while ensuring connectivity
    connections = ensure_sparse_connectivity(connections)
    
    return connections

def project_synthetic_to_original(x_syn, features):
    """Project synthetic nodes into original space using feature similarity"""
    sim_matrix = torch.cdist(x_syn, torch.tensor(features).float())
    weights = torch.softmax(-sim_matrix, dim=1)
    return weights

def threshold_sparse(matrix, sparsity=0.7):
    """Threshold matrix to achieve desired sparsity"""
    values = matrix.reshape(-1)
    threshold = torch.quantile(values, sparsity)
    return matrix * (matrix > threshold)

def ensure_sparse_connectivity(adj_matrix):
    """Ensure graph is connected while maintaining sparsity"""
    # Get minimum spanning tree
    edge_weights = -adj_matrix.cpu().numpy()
    mst = minimum_spanning_tree(edge_weights).toarray()
    
    # Add MST edges to ensure connectivity
    sparse_adj = threshold_sparse(adj_matrix)
    connected_adj = ((sparse_adj > 0) | torch.tensor(mst + mst.T > 0)).float()
    
    return connected_adj

def extend_spectral_properties(orig_eigenvals, orig_eigenvecs, A_aug, n_old, n_new):
    """
    Extend spectral properties using perturbation theory
    
    Parameters:
    - orig_eigenvals: Original eigenvalues (n_old,)
    - orig_eigenvecs: Original eigenvectors (n_old x n_old)
    - A_aug: Augmented adjacency matrix ((n_old + n_new) x (n_old + n_new))
    - n_old: Number of original nodes (12)
    - n_new: Number of new nodes (2)
    """
    # Convert to normalized Laplacian space
    L_aug = compute_normalized_laplacian(A_aug)
    
    # Extract perturbation blocks
    L11 = L_aug[:n_old, :n_old]  # (12 x 12)
    L12 = L_aug[:n_old, n_old:]  # (12 x 2)
    L21 = L_aug[n_old:, :n_old]  # (2 x 12)
    L22 = L_aug[n_old:, n_old:]  # (2 x 2)
    
    # Initialize augmented matrices
    n_total = n_old + n_new  # 14
    aug_eigenvals = torch.zeros(n_total)
    aug_eigenvecs = torch.zeros((n_total, n_total))
    
    # Copy original values
    aug_eigenvals[:n_old] = orig_eigenvals
    aug_eigenvecs[:n_old, :n_old] = orig_eigenvecs
    
    # First order eigenvalue corrections for original eigenvalues
    for i in range(n_old):
        v_i = orig_eigenvecs[:, i]
        correction = torch.sum(L12 @ L21 @ v_i * v_i)
        aug_eigenvals[i] += correction
    
    # Compute new eigenvalues for added nodes
    for i in range(n_new):
        # Estimate eigenvector for new node
        v_est = torch.zeros(n_old + n_new)
        
        # Project using connections to original nodes
        proj = L21[i] @ orig_eigenvecs
        v_est[:n_old] = proj
        v_est[n_old + i] = 1.0  # Set component for this new node
        
        # Normalize
        v_est = v_est / (torch.norm(v_est) + 1e-8)
        
        # Place in augmented eigenvector matrix
        aug_eigenvecs[:, n_old + i] = v_est
        
        # Compute corresponding eigenvalue using Rayleigh quotient
        Lv = L_aug @ v_est
        aug_eigenvals[n_old + i] = (v_est @ Lv) / (v_est @ v_est + 1e-8)
    
    # Orthogonalize the complete eigenvector matrix
    aug_eigenvecs = gram_schmidt(aug_eigenvecs)
    
    # Sort eigenvalues and eigenvectors
    idx = torch.argsort(aug_eigenvals)
    aug_eigenvals = aug_eigenvals[idx]
    aug_eigenvecs = aug_eigenvecs[:, idx]
    
    return aug_eigenvals, aug_eigenvecs

def compute_normalized_laplacian(A):
    """
    Compute normalized Laplacian matrix
    L = I - D^(-1/2)AD^(-1/2)
    """
    # Get degree matrix
    degrees = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-8))
    
    # Compute normalized Laplacian
    L = torch.eye(A.shape[0]).to(A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def gram_schmidt(vectors):
    """
    Perform Gram-Schmidt orthogonalization
    """
    basis = []
    for v in vectors.T:
        w = v.clone()
        for b in basis:
            w -= (w @ b) * b
        norm = torch.norm(w)
        if norm > 1e-8:  # Check if vector is non-zero
            w = w / norm
            basis.append(w)
        else:
            # If linear dependent, create random orthogonal vector
            w = torch.randn_like(v)
            for b in basis:
                w -= (w @ b) * b
            w = w / torch.norm(w)
            basis.append(w)
    
    return torch.stack(basis, dim=1)

# # Usage example:
# """
# aug_eigenvals, aug_eigenvecs = extend_spectral_properties(
#     orig_eigenvals,  # Original eigenvalues
#     orig_eigenvecs,  # Original eigenvectors
#     A_aug,           # Augmented adjacency matrix
#     n_old=12,        # Number of original nodes
#     n_new=k          # Number of new nodes
# )
# """

def augment_graph_louvain(residual, features, x_syn, A_distilled, num_new_nodes=18, orig_eigenvals=None, orig_eigenvecs=None):
    """
    Modified to select num_new_nodes clusters with highest Frobenius norm contribution
    """
    # Convert tensors to numpy
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
        if orig_eigenvals is not None and orig_eigenvecs is not None:
            return cluster_labels, A_distilled, x_syn, orig_eigenvals, orig_eigenvecs
        return cluster_labels, A_distilled, x_syn
    
    print(f"\nCreating {k} new synthetic nodes from top contributing clusters:")
    for cluster in selected_clusters:
        print(f"Cluster {cluster['label']}: {cluster['contribution']:.2f}% contribution")
    
    # Create augmented features
    new_features = torch.tensor(np.stack([c['features'] for c in selected_clusters])).float()
    x_aug = torch.vstack([x_syn, new_features])
    
    # Create augmented adjacency matrix
    n_syn = len(x_syn)
    A_aug = torch.zeros((n_syn + k, n_syn + k))
    
    # Keep existing synthetic node connections
    A_aug[:n_syn, :n_syn] = A_distilled
    
    # Add connections for new nodes
    cluster_connections = np.stack([c['connectivity'] for c in selected_clusters])
    
    new_to_existing = compute_new_connections(cluster_connections, x_syn, features)
    A_aug[n_syn:, :n_syn] = new_to_existing
    A_aug[:n_syn, n_syn:] = new_to_existing.T
    
    new_to_new = compute_new_to_new_connections(cluster_connections, k)
    A_aug[n_syn:, n_syn:] = new_to_new
    
    # Extend spectral properties if provided
    if orig_eigenvals is not None and orig_eigenvecs is not None:
        aug_eigenvals, aug_eigenvecs = extend_spectral_properties(
            orig_eigenvals,
            orig_eigenvecs,
            A_aug,
            n_old=n_syn,
            n_new=k
        )
        return cluster_labels, A_aug, x_aug, aug_eigenvals, aug_eigenvecs
    
    return cluster_labels, A_aug, x_aug

def compute_augmented_reconstruction(aug_eigenvals, aug_eigenvecs, L_eigenvectors, A_torch, k, threshold=0.01):
    """
    Compute augmented distilled matrix reconstruction and analysis
    
    Args:
        aug_eigenvals (torch.Tensor): Augmented eigenvalues
        aug_eigenvecs (torch.Tensor): Augmented eigenvectors
        L_eigenvectors (numpy.ndarray): Original Laplacian eigenvectors
        A_torch (torch.Tensor): Original adjacency matrix as torch tensor
        k (int): Number of eigenvectors to use
        threshold (float): Threshold for sparsification
        
    Returns:
        tuple: (aug_A_distilled, aug_R_sparsified, results) containing:
            - aug_A_distilled: Augmented distilled adjacency matrix
            - aug_R_sparsified: Sparsified residual matrix
            - results: Dictionary with analysis results
    """
    # Step 1: Create diagonal matrix with 1 - eigenvalues
    aug_diagonal_matrix = torch.diag(1 - aug_eigenvals)

    # Step 2: Compute normalized A_distilled
    aug_A_distilled = torch.mm(torch.mm(aug_eigenvecs, aug_diagonal_matrix), aug_eigenvecs.T)
    print(f"Augmented A_distilled shape: {aug_A_distilled.shape}")

    # Step 3: Extract the top k eigenvectors
    V = L_eigenvectors[:, :k]  # First k eigenvectors
    V = torch.tensor(V, dtype=torch.float64)
    print(f"V shape: {V.shape}")

    # Step 4: Ensure correct dtype and compute reconstruction
    aug_A_distilled = aug_A_distilled.to(dtype=torch.float64)
    aug_A_reconstructed = V @ aug_A_distilled @ V.T

    # Step 5: Compute residual
    aug_R = A_torch - aug_A_reconstructed
    print(f"Residual shape: {aug_R.shape}")

    # Step 6: Sparsify the residual matrix
    aug_R_sparsified = aug_R.clone()
    aug_R_sparsified[(aug_R_sparsified > -threshold) & (aug_R_sparsified < threshold)] = 0

    # Step 7: Compute statistics
    num_nonzero = torch.count_nonzero(aug_R_sparsified).item()
    num_zero = aug_R_sparsified.numel() - num_nonzero
    frobenius_norm = torch.norm(aug_R_sparsified, p='fro')
    sparsification_ratio = (num_zero / aug_R_sparsified.numel()) * 100

    # Print results
    print(f"\nSparsification Results (threshold = {threshold}):")
    print(f"Number of nonzero elements: {num_nonzero}")
    print(f"Number of zero elements: {num_zero}")
    print(f"Frobenius norm of R_sparsified: {frobenius_norm}")
    print(f"Sparsification ratio: {sparsification_ratio:.2f}%")

    # Prepare results dictionary
    results = {
        'nonzero_count': num_nonzero,
        'zero_count': num_zero,
        'frobenius_norm': frobenius_norm,
        'sparsification_ratio': sparsification_ratio,
        'aug_A_reconstructed': aug_A_reconstructed,
        'aug_R': aug_R
    }

    return aug_A_distilled, aug_R_sparsified, results



def reassign_augmented_graph_labels(x_aug, A_aug, eigenvecs_aug, x_train, y_train, reduction_rate, alpha=0.5):
    """
    Assign labels to all nodes in augmented distilled graph
    
    Args:
        x_aug (torch.Tensor): Features of all nodes in augmented graph
        A_aug (torch.Tensor): Augmented adjacency matrix
        eigenvecs_aug (torch.Tensor): Augmented eigenvectors
        x_train (torch.Tensor): Training node features
        y_train (torch.Tensor): Training node labels
        reduction_rate (float): Target size relative to original graph
        alpha (float): Weight between feature and structural similarity (0-1)
        
    Returns:
        torch.Tensor: Labels for all nodes in augmented graph
    """
    # Calculate target number of nodes per class
    n_total = len(y_train)
    class_counts = Counter(y_train.cpu().numpy())
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    n_aug = len(x_aug)
    
    # Calculate target distribution for augmented graph
    target_counts = {}
    running_sum = 0
    for i, (c, count) in enumerate(sorted_classes):
        if i == len(sorted_classes) - 1:
            target_counts[c] = n_aug - running_sum
        else:
            # Scale up the minimum nodes per class for augmented size
            min_nodes = max(int(count * reduction_rate), 1)
            aug_scale = n_aug / int(n_total * reduction_rate)
            target_counts[c] = max(int(min_nodes * aug_scale), 1)
            running_sum += target_counts[c]
    
    # Compute combined similarity scores
    feat_sim = F.normalize(torch.mm(x_aug, x_train.t()), dim=1)
    
    struct_embeddings = eigenvecs_aug @ A_aug
    train_embeddings = eigenvecs_aug[:len(y_train)] @ torch.eye(len(y_train), device=y_train.device)
    struct_sim = F.normalize(torch.mm(struct_embeddings, train_embeddings.t()), dim=1)
    
    combined_sim = alpha * feat_sim + (1 - alpha) * struct_sim
    
    # Initialize tensors to store results
    assigned_labels = torch.zeros(n_aug, dtype=torch.long, device=x_aug.device)
    assigned_mask = torch.zeros(n_aug, dtype=torch.bool, device=x_aug.device)
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
