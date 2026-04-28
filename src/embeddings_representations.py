



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler






def analyze_factor_importance(complete_results, movie_id_to_title, 
                              index_to_movie, output_dir='xai_analysis1'):
    """
    Identify what each latent dimension represents
    """
    item_factors = complete_results['model_parameters']['item_factors']
    K = complete_results['hyperparameters']['K']
    

   
    for k in range(min(K, 5)):  
        print(f"\n--- Dimension {k} ---")
        
      
        loadings = item_factors[:, k]
        
        # Top 10 positive
        top_idx = np.argsort(loadings)[-10:][::-1]
        print(f"\nHighest loadings (positive):")
        for rank, idx in enumerate(top_idx, 1):
            movie_id = index_to_movie[idx]
            title = movie_id_to_title.get(movie_id, "Unknown")
            print(f"  {rank}. {title[:50]:<50} ({loadings[idx]:+.3f})")
        
        #
        bottom_idx = np.argsort(loadings)[:10]
        print(f"\nLowest loadings (negative):")
        for rank, idx in enumerate(bottom_idx, 1):
            movie_id = index_to_movie[idx]
            title = movie_id_to_title.get(movie_id, "Unknown")
            print(f"  {rank}. {title[:50]:<50} ({loadings[idx]:+.3f})")
    
   
    factor_variances = np.var(item_factors, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(K), factor_variances, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Latent Dimension', fontsize=10, fontweight='bold')
    ax.set_ylabel('Variance Explained', fontsize=10, fontweight='bold')
    ax.set_title('Importance of Each Latent Dimension', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_factor_importance.pdf', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/02_factor_importance.pdf")
def select_optimal_dimensions(embeddings, name="Embeddings", 
                              variance_threshold=0.80,
                              k_min=5, k_max=15):
    """
    Automatically selects optimal number of dimensions
    based on cumulative variance
    """
    # Calculate variance per dimension
    variances = np.var(embeddings, axis=0)
    sorted_indices = np.argsort(variances)[::-1]
    sorted_variances = variances[sorted_indices]
    
    # Cumulative variance
    cumsum_var = np.cumsum(sorted_variances)
    total_var = np.sum(variances)
    pct_cumsum = cumsum_var / total_var
    
    # Find optimal k (first dimension where threshold is exceeded)
    k_optimal = np.argmax(pct_cumsum >= variance_threshold) + 1
    k_optimal = max(k_min, min(k_optimal, k_max))  # Apply constraints
    
    variance_captured = pct_cumsum[k_optimal - 1] * 100
    
   
    for i in range(k_optimal):
        idx = sorted_indices[i]
        print(f"  Dim {idx:2d}: var={variances[idx]:.5f} ({variances[idx]/total_var*100:.2f}%)")
    
    return sorted_indices[:k_optimal], k_optimal, variance_captured



def apply_pca_with_diagnostics(embeddings, selected_dims, name="Embeddings"):
    """
    Applies PCA on selected dimensions with complete diagnostics
    """
  
    embeddings_selected = embeddings[:, selected_dims]
    
  
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_selected)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_scaled)
    

    

    print(f"\n  Contributions to principal components:")
    components = pca.components_
    for i, comp in enumerate(components):
        top_contrib = np.argsort(np.abs(comp))[-3:][::-1]  # Top 3 contributions
        print(f"    PC{i+1} - Top contributors:", end=" ")
        for idx in top_contrib:
            print(f"Dim{selected_dims[idx]}({comp[idx]:.2f})", end=" ")
        print()
    
    return embeddings_2d, pca
