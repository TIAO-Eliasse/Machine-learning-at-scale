
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
from prediction import fit_new_user, predict_all_for_user_without_user_bias

import csv
from collections import defaultdict

class NewUserXAI:
    """
    Explainable AI for a new user profile based on LOTR rating
    """
    
    def __init__(self, complete_results, movie_id_to_title, index_to_movie, 
                 movie_to_index, K, lambda_reg, tau_reg, output_dir='xai_new_user'):
        self.complete_results = complete_results
        self.movie_id_to_title = movie_id_to_title
        self.index_to_movie = index_to_movie
        self.movie_to_index = movie_to_index
        self.K = K
        self.lambda_reg = lambda_reg
        self.tau_reg = tau_reg
        self.output_dir = output_dir
        
       
        self.item_factors = complete_results['model_parameters']['item_factors']
        self.item_biases = complete_results['model_parameters']['item_biases']
        self.N = complete_results['hyperparameters']['N']
        
        os.makedirs(output_dir, exist_ok=True)
        
     
        self._find_lotr()
        

        self.user_factor, self.user_bias = None, None
        
        print("="*80)
        print("XAI FOR NEW USER WHO LOVES LORD OF THE RINGS")
        print("="*80)
    
    def _find_lotr(self):
        """Find LOTR movie"""
        for movie_id, title in self.movie_id_to_title.items():
            if "Fellowship of the Ring" in title:
                if movie_id in self.movie_to_index:
                    self.lotr_item_idx = self.movie_to_index[movie_id]
                    self.lotr_title = title
                    break
    
    def fit_new_user_profile(self, lotr_rating=5.0):
        """
        Fit new user profile based on LOTR rating
        """
        print(f"\nFitting new user profile:")
        print(f"  User rates '{self.lotr_title}' with {lotr_rating} stars")
        
       
        user_ratings = [(self.lotr_item_idx, lotr_rating)]
        self.user_factor, self.user_bias = fit_new_user(
            user_ratings,
            self.item_factors,
            self.item_biases,
            self.K,
            self.lambda_reg,
            self.tau_reg
        )
        
        print(f"  Trained user bias: {self.user_bias:+.4f}")
        print(f"  User factor magnitude: {np.linalg.norm(self.user_factor):.4f}")
     
        lotr_factor = self.item_factors[self.lotr_item_idx]
        alignment = np.dot(self.user_factor, lotr_factor) / (
            np.linalg.norm(self.user_factor) * np.linalg.norm(lotr_factor)
        )
        print(f"  Alignment with LOTR: {alignment:+.4f}")
    
    def get_top_recommendations(self, beta=1.0, top_n=10):
        """
        Get top N recommendations for new user
        """
        if self.user_factor is None:
            raise ValueError("Must fit user profile first!")
        
     
        all_predictions = predict_all_for_user_without_user_bias(
            self.user_factor,
            self.item_factors,
            self.user_bias,
            self.item_biases,
            beta
        )
        
       
        top_indices = np.argsort(all_predictions)[::-1][:top_n]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            movie_id = self.index_to_movie[idx]
            title = self.movie_id_to_title.get(movie_id, "Unknown")
            
        
            interaction = np.sum(self.user_factor * self.item_factors[idx])
            bias_contrib = beta * self.item_biases[idx]
            prediction = all_predictions[idx]
            
        
            alignment = np.dot(self.user_factor, self.item_factors[idx]) / (
                np.linalg.norm(self.user_factor) * np.linalg.norm(self.item_factors[idx])
            )
            
            recommendations.append({
                'rank': rank,
                'item_idx': idx,
                'title': title,
                'prediction': prediction,
                'interaction': interaction,
                'bias_contrib': bias_contrib,
                'alignment': alignment,
                'dimension_contributions': self.user_factor * self.item_factors[idx]
            })
        
        return recommendations
    
    def explain_top_recommendations(self, beta=1.0, top_n=10):
        """
        Comprehensive explanation of top recommendations
        """
    
        recs = self.get_top_recommendations(beta, top_n)
        
        print(f"\n{'Rank':<6}{'Title':<50}{'Pred':<8}{'u·v':<10}{'β×b_i':<10}{'Align':<8}")
        print("-"*100)
        
        for rec in recs:
            title_short = rec['title'][:47] + "..." if len(rec['title']) > 50 else rec['title']
            print(f"{rec['rank']:<6}{title_short:<50}{rec['prediction']:<8.3f}"
                  f"{rec['interaction']:<10.3f}{rec['bias_contrib']:<10.3f}{rec['alignment']:<8.3f}")
        
        return recs
    
    def visualize_recommendations_explainability(self, beta=1.0, top_n=10):
        """
        Create comprehensive XAI visualization for recommendations
        """
        recs = self.get_top_recommendations(beta, top_n)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
      
        

        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_recommendations_waterfall(ax1, recs[:5], beta)
        
     
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_alignment_scatter(ax2, recs)
        
       
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_component_breakdown(ax3, recs, beta)
       
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_statistics_box(ax4, recs, beta)
        

        ax5 = fig.add_subplot(gs[2, :])
        self._plot_dimension_heatmap(ax5, recs)
        
        plt.suptitle(f'Explainable AI: Top {top_n} Recommendations for New User (LOTR Fan)\nβ={beta}',
                     fontsize=14, fontweight='bold')
        
        filename = f'{self.output_dir}/new_user_recommendations_xai_beta{beta}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {filename}")
        plt.close()
    
    def _plot_recommendations_waterfall(self, ax, recs, beta):
        """Waterfall chart for top recommendations"""
        n_movies = len(recs)
        x_pos = np.arange(n_movies)
        width = 0.35
        
        interactions = [r['interaction'] for r in recs]
        bias_contribs = [r['bias_contrib'] for r in recs]
        predictions = [r['prediction'] for r in recs]
        
      
        ax.bar(x_pos - width/2, interactions, width, label='Interaction (u·v)',
              color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        ax.bar(x_pos + width/2, bias_contribs, width, label=f'Bias (β={beta})',
              color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1)
        
  
        ax.plot(x_pos, predictions, 'ro-', linewidth=2.5, markersize=8,
               label='Total Prediction', zorder=10)

        for i, (inter, bias, pred) in enumerate(zip(interactions, bias_contribs, predictions)):
            ax.text(i, pred + 0.1, f'{pred:.2f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold', color='red')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"#{r['rank']}" for r in recs], fontsize=10)
        ax.set_xlabel('Recommendation Rank', fontweight='bold')
        ax.set_ylabel('Contribution', fontweight='bold')
        ax.set_title('Prediction Decomposition (Top 5)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    def _plot_alignment_scatter(self, ax, recs):
        """Scatter plot of alignment vs prediction"""
        alignments = [r['alignment'] for r in recs]
        predictions = [r['prediction'] for r in recs]
        ranks = [r['rank'] for r in recs]
        
        scatter = ax.scatter(alignments, predictions, s=200, c=ranks,
                           cmap='RdYlGn_r', alpha=0.7, edgecolors='black',
                           linewidths=1.5)
        
   
        for r in recs:
            ax.text(r['alignment'], r['prediction'], str(r['rank']),
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('User-Item Alignment', fontweight='bold')
        ax.set_ylabel('Predicted Rating', fontweight='bold')
        ax.set_title('Alignment vs Prediction', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Rank', fontweight='bold')
    
    def _plot_component_breakdown(self, ax, recs, beta):
        """Horizontal bar chart of components"""
        titles = [r['title'][:30] + "..." if len(r['title']) > 30 else r['title'] 
                 for r in recs]
        interactions = [r['interaction'] for r in recs]
        bias_contribs = [r['bias_contrib'] for r in recs]
        
        y_pos = np.arange(len(titles))
        width = 0.35
        
        ax.barh(y_pos - width/2, interactions, width, label='Interaction',
               color='#3498db', alpha=0.8, edgecolor='black')
        ax.barh(y_pos + width/2, bias_contribs, width, label='Bias',
               color='#e67e22', alpha=0.8, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Contribution Value', fontweight='bold')
        ax.set_title('Component Breakdown', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    def _plot_statistics_box(self, ax, recs, beta):
        """Summary statistics"""
        ax.axis('off')
        
        mean_pred = np.mean([r['prediction'] for r in recs])
        mean_inter = np.mean([r['interaction'] for r in recs])
        mean_bias = np.mean([r['bias_contrib'] for r in recs])
        mean_align = np.mean([r['alignment'] for r in recs])
        
        stats_text = f"""RECOMMENDATION SUMMARY
{'='*30}

Top {len(recs)} Movies Statistics:

Mean Prediction: {mean_pred:.3f}
Mean Interaction: {mean_inter:+.3f}
Mean Bias: {mean_bias:+.3f}
Mean Alignment: {mean_align:+.3f}

β value: {beta}

Top Movie:
  {recs[0]['title'][:25]}
  Pred: {recs[0]['prediction']:.3f}
  Align: {recs[0]['alignment']:+.3f}
"""
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
    
    def _plot_dimension_heatmap(self, ax, recs):
        """Heatmap of dimension contributions"""
     
        dim_matrix = np.array([r['dimension_contributions'] for r in recs])
        
    
        im = ax.imshow(dim_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.max(np.abs(dim_matrix)), 
                      vmax=np.max(np.abs(dim_matrix)))
        
        ax.set_xlabel('Latent Dimension k', fontweight='bold')
        ax.set_ylabel('Recommendation Rank', fontweight='bold')
        ax.set_title('Per-Dimension Contributions (u_k × v_k)', fontweight='bold')
        
   
        ax.set_yticks(range(len(recs)))
        ax.set_yticklabels([f"#{r['rank']}" for r in recs])
        

        ax.set_xticks(range(0, self.K, 2))
        
   
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Contribution', fontweight='bold')
    
    def compare_beta_impact(self, beta_values=[0.0, 0.5, 1.0], top_n=10):
        """
        Compare recommendations across different beta values
        """
        print("\n" + "="*80)
        print(f"COMPARING BETA IMPACT ON TOP {top_n} RECOMMENDATIONS")
        print("="*80)
        
        results_by_beta = {}
        
        for beta in beta_values:
            recs = self.get_top_recommendations(beta, top_n)
            results_by_beta[beta] = recs
            
          
        self._visualize_beta_comparison(results_by_beta, beta_values, top_n)
        
        return results_by_beta
    
    def _visualize_beta_comparison(self, results_by_beta, beta_values, top_n):
        """Visualize beta comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        

        ax = axes[0, 0]
        for beta in beta_values:
            preds = [r['prediction'] for r in results_by_beta[beta]]
            ax.plot(range(1, top_n+1), preds, 'o-', linewidth=2, markersize=6,
                   label=f'β={beta}', alpha=0.8)
        
        ax.set_xlabel('Recommendation Rank', fontweight='bold')
        ax.set_ylabel('Predicted Rating', fontweight='bold')
        ax.set_title('Predictions by Rank', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        
        ax = axes[0, 1]
        x = np.arange(len(beta_values))
        width = 0.35
        
        mean_inter = [np.mean([r['interaction'] for r in results_by_beta[b]]) 
                     for b in beta_values]
        mean_bias = [np.mean([r['bias_contrib'] for r in results_by_beta[b]]) 
                    for b in beta_values]
        
        ax.bar(x - width/2, mean_inter, width, label='Mean Interaction',
              color='#3498db', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, mean_bias, width, label='Mean Bias',
              color='#e67e22', alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'β={b}' for b in beta_values])
        ax.set_ylabel('Mean Contribution', fontweight='bold')
        ax.set_title('Component Contributions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
       
        ax = axes[1, 0]
        top_movies = {}
        for beta in beta_values:
            top_title = results_by_beta[beta][0]['title'][:25]
            top_pred = results_by_beta[beta][0]['prediction']
            top_movies[f'β={beta}'] = (top_title, top_pred)
        
        bars = ax.bar(range(len(beta_values)), 
                     [top_movies[f'β={b}'][1] for b in beta_values],
                     color='steelblue', alpha=0.8, edgecolor='black')
        
        ax.set_xticks(range(len(beta_values)))
        ax.set_xticklabels([f'β={b}' for b in beta_values])
        ax.set_ylabel('Prediction Score', fontweight='bold')
        ax.set_title('Top Recommendation Score', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
     
        ax = axes[1, 1]
   
        all_top5_indices = set()
        for beta in beta_values:
            for r in results_by_beta[beta][:5]:
                all_top5_indices.add(r['item_idx'])
        
        
        for item_idx in list(all_top5_indices)[:5]: 
            ranks = []
            for beta in beta_values:
                rank = next((r['rank'] for r in results_by_beta[beta] if r['item_idx'] == item_idx), top_n+1)
                ranks.append(rank)
            
            movie_title = self.movie_id_to_title.get(self.index_to_movie[item_idx], "Unknown")[:15]
            ax.plot(beta_values, ranks, 'o-', linewidth=2, markersize=6,
                   label=movie_title, alpha=0.8)
        
        ax.set_xlabel('β Value', fontweight='bold')
        ax.set_ylabel('Rank', fontweight='bold')
        ax.set_title('Rank Stability Across β', fontweight='bold')
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Beta Comparison: Impact on Top {top_n} Recommendations',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.output_dir}/beta_comparison_analysis.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {filename}")
        plt.close()



def analyze_polarizing_movies(results, movie_to_index, index_to_movie, 
                              user_ratings_train, user_items_train, user_start_end_train,
                              movies_file_path, output_dir='polarization_analysis'):
    """
    Comprehensive analysis of movie polarization and its impact on predictions
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    item_factors = results.get('item_factors')
    item_biases = results.get('item_biases')
    user_factors = results.get('user_factors')
    
    N = item_factors.shape[0]
    K = item_factors.shape[1]
    
    movie_id_to_title = {}
    with open(movies_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            movie_id_to_title[row[0]] = row[1]
    
   
    
    magnitudes = np.zeros(N, dtype=np.float32)
    for n in range(N):
        norm_squared = 0.0
        for k in range(K):
            norm_squared += item_factors[n, k] ** 2
        magnitudes[n] = np.sqrt(norm_squared)
    
    rating_counts = np.zeros(N, dtype=np.int32)
    rating_means = np.zeros(N, dtype=np.float32)
    rating_stds = np.zeros(N, dtype=np.float32)
    
    movie_ratings_dict = defaultdict(list)
    
    for user_idx in range(len(user_start_end_train)):
        start, end = user_start_end_train[user_idx]
        for i in range(start, end):
            item_idx = user_items_train[i]
            rating = user_ratings_train[i]
            movie_ratings_dict[item_idx].append(rating)
    
    for item_idx in range(N):
        if item_idx in movie_ratings_dict:
            ratings = np.array(movie_ratings_dict[item_idx])
            rating_counts[item_idx] = len(ratings)
            rating_means[item_idx] = np.mean(ratings)
            rating_stds[item_idx] = np.std(ratings)
    
    top_n = 20
    min_ratings = 50
    valid_mask = rating_counts >= min_ratings
    valid_indices = np.where(valid_mask)[0]
    
    valid_magnitudes = magnitudes[valid_indices]
    sorted_idx = np.argsort(valid_magnitudes)[::-1][:top_n]
    top_polarizing_indices = valid_indices[sorted_idx]
    
    print(f"\nTop {top_n} Most Polarizing Movies (min {min_ratings} ratings):\n")
    print(f"{'Rank':<6}{'Title':<50}{'|v|':<10}{'Bias':<10}{'Ratings':<10}{'Mean':<10}{'Std':<10}")
    print("-"*110)
    
    polarizing_data = []
    
    for rank, idx in enumerate(top_polarizing_indices, 1):
        movie_id = index_to_movie[idx]
        title = movie_id_to_title.get(movie_id, "Unknown")
        
        data = {
            'rank': rank,
            'index': idx,
            'title': title[:47] + "..." if len(title) > 50 else title,
            'magnitude': magnitudes[idx],
            'bias': item_biases[idx],
            'count': rating_counts[idx],
            'mean': rating_means[idx],
            'std': rating_stds[idx],
            'factors': item_factors[idx]
        }
        polarizing_data.append(data)
        
        print(f"{rank:<6}{data['title']:<50}{data['magnitude']:<10.4f}{data['bias']:<10.4f}"
              f"{data['count']:<10}{data['mean']:<10.2f}{data['std']:<10.2f}")
 
    
    valid_mags = magnitudes[valid_mask]
    valid_counts = rating_counts[valid_mask]
    valid_stds = rating_stds[valid_mask]
    valid_means = rating_means[valid_mask]
    valid_biases = item_biases[valid_mask]
    
    corr_mag_count = np.corrcoef(valid_mags, valid_counts)[0, 1]
    corr_mag_std = np.corrcoef(valid_mags, valid_stds)[0, 1]
    corr_mag_mean = np.corrcoef(valid_mags, valid_means)[0, 1]
    corr_mag_bias = np.corrcoef(valid_mags, valid_biases)[0, 1]
    
    
    if corr_mag_count > 0.3:
        print("  - High correlation with rating count: Popular movies tend to be polarizing")
    if corr_mag_std > 0.3:
        print("  - High correlation with rating variance: Divisive opinions create polarization")
    if abs(corr_mag_bias) > 0.3:
        print(f"  - {'Positive' if corr_mag_bias > 0 else 'Negative'} correlation with bias: "
              f"{'Well-liked' if corr_mag_bias > 0 else 'Poorly-rated'} movies are more polarizing")

    
    num_example_users = 5
    example_user_indices = np.random.choice(len(user_start_end_train), num_example_users, replace=False)
    
    print("\nPrediction Analysis for Sample Users:\n")
    
    for user_idx in example_user_indices:
        user_factor = user_factors[user_idx]
        predictions = []
        
        for movie_data in polarizing_data[:5]:
            item_idx = movie_data['index']
            dot_product = 0.0
            for k in range(K):
                dot_product += user_factor[k] * item_factors[item_idx, k]
            
            prediction = item_biases[item_idx] + dot_product
            
            start, end = user_start_end_train[user_idx]
            actual_rating = None
            for i in range(start, end):
                if user_items_train[i] == item_idx:
                    actual_rating = user_ratings_train[i]
                    break
            
            predictions.append({
                'title': movie_data['title'],
                'prediction': prediction,
                'actual': actual_rating,
                'bias': item_biases[item_idx],
                'dot_product': dot_product
            })
        
        print(f"\nUser {user_idx}:")
        print(f"  {'Movie':<40}{'Pred':<8}{'Actual':<8}{'Bias':<8}{'u·v':<8}")
        print("  " + "-"*70)
        for pred in predictions:
            actual_str = f"{pred['actual']:.2f}" if pred['actual'] is not None else "N/A"
            print(f"  {pred['title']:<40}{pred['prediction']:<8.2f}{actual_str:<8}"
                  f"{pred['bias']:<8.2f}{pred['dot_product']:<8.2f}")
    
    create_polarization_visualizations(
        magnitudes, rating_counts, rating_stds, rating_means, item_biases,
        valid_mask, polarizing_data, output_dir
    )
    
    return polarizing_data, {
        'magnitudes': magnitudes,
        'rating_counts': rating_counts,
        'rating_stds': rating_stds,
        'rating_means': rating_means,
        'correlations': {
            'mag_count': corr_mag_count,
            'mag_std': corr_mag_std,
            'mag_mean': corr_mag_mean,
            'mag_bias': corr_mag_bias
        }
    }


def create_polarization_visualizations(magnitudes, rating_counts, rating_stds, 
                                       rating_means, item_biases, valid_mask,
                                       polarizing_data, output_dir):
    """

    """
    
    valid_mags = magnitudes[valid_mask]
    valid_counts = rating_counts[valid_mask]
    valid_stds = rating_stds[valid_mask]
    valid_biases = item_biases[valid_mask]
    
    top_indices = [d['index'] for d in polarizing_data[:10]]
    top_counts = rating_counts[top_indices]
    top_mags = magnitudes[top_indices]
    top_stds = rating_stds[top_indices]
    top_biases = item_biases[top_indices]
    

    
    fig_combined, axes = plt.subplots(1, 3, figsize=(8, 3))
    
    FONTSIZE_LABEL = 9
    FONTSIZE_TITLE = 10
    FONTSIZE_TICK = 8
    FONTSIZE_LEGEND = 7
    
    ax = axes[0]
    ax.scatter(valid_counts, valid_mags, alpha=0.3, s=15, c='steelblue')
    ax.scatter(top_counts, top_mags, alpha=0.8, s=60, c='red', 
              marker='*', edgecolors='black', linewidths=0.8, label='Top 10')
    
    corr = np.corrcoef(valid_counts, valid_mags)[0, 1]
    ax.set_xlabel('Number of Ratings', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Latent Vector Magnitude |v|', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title(f'Polarization vs Popularity (r={corr:.3f})', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.tick_params(labelsize=FONTSIZE_TICK)

    ax = axes[1]
    ax.scatter(valid_stds, valid_mags, alpha=0.3, s=15, c='steelblue')
    ax.scatter(top_stds, top_mags, alpha=0.8, s=60, c='red', 
              marker='*', edgecolors='black', linewidths=0.8, label='Top 10')
    
    corr = np.corrcoef(valid_stds, valid_mags)[0, 1]
    ax.set_xlabel('Rating Standard Deviation', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Latent Vector Magnitude |v|', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title(f'Polarization vs Variance (r={corr:.3f})', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.tick_params(labelsize=FONTSIZE_TICK)
  
    ax = axes[2]
    ax.scatter(valid_biases, valid_mags, alpha=0.3, s=15, c='steelblue')
    ax.scatter(top_biases, top_mags, alpha=0.8, s=60, c='red', 
              marker='*', edgecolors='black', linewidths=0.8, label='Top 10')
    
    corr = np.corrcoef(valid_biases, valid_mags)[0, 1]
    ax.set_xlabel('Item Bias', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Latent Vector Magnitude |v|', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title(f'Polarization vs Quality (r={corr:.3f})', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    
    plt.tight_layout()
    fig_combined.savefig(f'{output_dir}/01_polarization_correlations.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/01_polarization_correlations.pdf")
    
  
    fig2, ax = plt.subplots(figsize=(8, 3))
    
    ax.hist(valid_mags, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    for mag in top_mags:
        ax.axvline(mag, color='red', alpha=0.5, linestyle='--', linewidth=1)
    
    ax.set_xlabel('Latent Vector Magnitude |v|', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title('Distribution of Movie Polarization', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    
    plt.tight_layout()
    fig2.savefig(f'{output_dir}/02_magnitude_distribution.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/02_magnitude_distribution.pdf")

    
    fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 3))
    
    top_5_polarizing = polarizing_data[:5]
    user_preference_range = np.linspace(-1, 1, 100)
    
    for movie_data in top_5_polarizing:
        predictions = []
        for user_pref in user_preference_range:
            dot_product = user_pref * movie_data['magnitude']
            prediction = movie_data['bias'] + dot_product
            predictions.append(prediction)
        
        ax_left.plot(user_preference_range, predictions, 
                    label=movie_data['title'][:20], linewidth=1.5, alpha=0.8)
    
    ax_left.set_xlabel('User-Item Alignment', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax_left.set_ylabel('Predicted Rating', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax_left.set_title('Prediction Sensitivity', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_left.legend(fontsize=6, loc='best')
    ax_left.grid(True, alpha=0.3, linestyle='--')
    ax_left.axhline(y=3, color='gray', linestyle=':', alpha=0.5)
    ax_left.tick_params(labelsize=FONTSIZE_TICK)
    
    ax_right.axis('off')
    
    summary_text = f"""KEY FINDINGS

Polarization Drivers:
 - Popular movies have more ratings
   leading to larger latent vectors
 - Divisive content shows high rating
   variance amplifying magnitude
 - Item bias captures base preference
   independent of personalization

Prediction Impact:
 - High |v|: Stronger influence of
   user-item alignment
 - Low |v|: Predictions rely more
   on item bias
 - Slope proportional to |v|,
   steeper for polarizing movies

Recommendation Quality:
 - Polarizing movies offer greater
   personalization potential
 - Better differentiation between
   user preferences
"""
    
    ax_right.text(0.05, 0.95, summary_text, transform=ax_right.transAxes,
                 fontsize=7, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=0.6))
    
    plt.tight_layout()
    fig3.savefig(f'{output_dir}/03_prediction_impact.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/03_prediction_impact.pdf")
    
    plt.close('all')

