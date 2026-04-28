import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from textwrap import wrap
from scipy import stats
import os
from typing import Dict, List, Tuple, Optional

from numba import njit, prange





@njit
def fit_new_user_optimized(movie_indices, ratings, item_factors, item_biases,
                          K, lambda_reg, tau_reg, num_iterations=50, eps=1e-8):
    """
    Optimized version using single loops and Numba JIT compilation.

    Parameters
    ----------
    movie_indices : np.array
        Array of movie indices that the user rated
    ratings : np.array
        Array of ratings given by the user
    item_factors : np.array
        Item factor matrix (N x K)
    item_biases : np.array
        Item bias vector
    K : int
        Number of latent factors
    lambda_reg, tau_reg : float
        Regularization parameters
    num_iterations : int
        Number of iterations for optimization
    eps : float
        Small constant for numerical stability

    Returns
    -------
    tuple
        (user_factor, user_bias)
    """
    user_bias = 0.0
    user_factor = np.random.normal(0, 0.1, K).astype(np.float32)

    n = len(movie_indices)
    I = np.eye(K, dtype=np.float32)

    # Extract relevant item factors and biases
    V_m = np.zeros((n, K), dtype=np.float32)
    b_i_m = np.zeros(n, dtype=np.float32)

    for i in range(n):
        idx = movie_indices[i]
        b_i_m[i] = item_biases[idx]
        for k in range(K):
            V_m[i, k] = item_factors[idx, k]

    for iteration in range(num_iterations):
        # Update user bias
        preds = np.zeros(n, dtype=np.float32)
        for i in range(n):
            # Compute dot product
            dot_prod = 0.0
            for k in range(K):
                dot_prod += V_m[i, k] * user_factor[k]
            preds[i] = dot_prod

        residuals_sum = 0.0
        for i in range(n):
            residuals_sum += ratings[i] - preds[i] - b_i_m[i]

        num = lambda_reg * residuals_sum
        den = lambda_reg * n + tau_reg
        user_bias = num / (den + eps)

        # Update user factor
        r_adj = np.zeros(n, dtype=np.float32)
        for i in range(n):
            r_adj[i] = ratings[i] - user_bias - b_i_m[i]

        # A = lambda_reg * (V_m.T @ V_m) + tau_reg * I + eps * I
        A = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                # Compute V_m.T @ V_m
                for idx in range(n):
                    A[i, j] += lambda_reg * V_m[idx, i] * V_m[idx, j]
                # Add regularization
                if i == j:
                    A[i, j] += tau_reg + eps

        # b = lambda_reg * (V_m.T @ r_adj)
        b = np.zeros(K, dtype=np.float32)
        for k in range(K):
            for idx in range(n):
                b[k] += lambda_reg * V_m[idx, k] * r_adj[idx]

        user_factor = np.linalg.solve(A, b)

    return user_factor, user_bias


def fit_new_user(user_ratings, item_factors, item_biases,
                 K, lambda_reg, tau_reg, num_iterations=50, eps=1e-8):
    """
    Wrapper function to prepare data and call optimized version.
    """
    movie_indices = np.array([idx for idx, rating in user_ratings], dtype=np.int32)
    ratings = np.array([rating for idx, rating in user_ratings], dtype=np.float32)

    return fit_new_user_optimized(
        movie_indices, ratings, item_factors, item_biases,
        K, lambda_reg, tau_reg, num_iterations, eps
    )

@njit
def predict_all_for_user_without_user_bias(user_factor, item_factors, user_bias, item_biases,weight):
    """
    Returns a vector of predictions for all items for a given user.
    Uses single loops for efficiency.

    Parameters
    ----------
    user_factor : np.array
        User factor vector (K,)
    item_factors : np.array
        Item factor matrix (N, K)
    user_bias : float
        User bias
    item_biases : np.array
        Item bias vector (N,)

    Returns
    -------
    np.array
        Predictions for all items
    """
    N = item_factors.shape[0]
    K = item_factors.shape[1]
    predictions = np.zeros(N, dtype=np.float32)

    for n in range(N):
        # Compute dot product
        dot_prod = 0.0
        for k in range(K):
            dot_prod += user_factor[k] * item_factors[n, k]
        #predictions[n] = dot_prod + user_bias + item_biases[n]
        predictions[n] = dot_prod +  weight*item_biases[n]
    return predictions


class MovieRecommendationVisualizer:
    """
    A class to visualize movie recommendations using different scoring methods.
    
    Attributes:
        output_dir (str): Directory to save plots
        movie_id_to_title (dict): Mapping from movie IDs to titles
        movie_to_index (dict): Mapping from movie IDs to indices
        index_to_movie (dict): Mapping from indices to movie IDs
        item_factors (np.ndarray): Item factor matrix
        item_biases (np.ndarray): Item bias vector
    """
    
    def __init__(self, 
                 movie_id_to_title: Dict,
                 movie_to_index: Dict,
                 index_to_movie: Dict,
                 item_factors: np.ndarray,
                 item_biases: np.ndarray,
                 output_dir: str = "recommendation_plots"):
        """
        Initialize the visualizer with movie data and model parameters.
        
        Args:
            movie_id_to_title: Dictionary mapping movie IDs to titles
            movie_to_index: Dictionary mapping movie IDs to matrix indices
            index_to_movie: Dictionary mapping matrix indices to movie IDs
            item_factors: Item factor matrix from the recommendation model
            item_biases: Item bias vector from the recommendation model
            output_dir: Directory to save generated plots
        """
        self.output_dir = output_dir
        self.movie_id_to_title = movie_id_to_title
        self.movie_to_index = movie_to_index
        self.index_to_movie = index_to_movie
        self.item_factors = item_factors
        self.item_biases = item_biases
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plot style
        self._setup_plot_style()
        
        # Define method colors
        self.method_colors = {
            'Method with β=0.05': '#3498DB',
            'Method with β=1': '#3498DB'
        }
    
    def _setup_plot_style(self):
        """Configure matplotlib style for publication-quality plots."""
        plt.style.use('default')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
    
    @staticmethod
    def wrap_title(title: str, max_width: int = 45) -> str:
        """
        Wrap long movie titles intelligently.
        
        Args:
            title: Movie title to wrap
            max_width: Maximum characters per line
            
        Returns:
            Wrapped title with newlines
        """
        if len(title) <= max_width:
            return title
        
        # Handle titles with year in parentheses
        if '(' in title and ')' in title:
            parts = title.split('(')
            if len(parts) == 2:
                main_title = parts[0].strip()
                year = '(' + parts[1]
                if len(main_title) <= max_width - 7:
                    return main_title + '\n' + year
        
        # Otherwise use textwrap
        wrapped = '\n'.join(wrap(title, width=max_width))
        return wrapped
    
    def find_movie(self, movie_title: str) -> Optional[Tuple[int, int]]:
        """
        Find a movie by title and return its ID and index.
        
        Args:
            movie_title: Title or partial title to search for
            
        Returns:
            Tuple of (movie_id, movie_index) or None if not found
        """
        for movie_id, title in self.movie_id_to_title.items():
            if movie_title in title:
                if movie_id in self.movie_to_index:
                    movie_index = self.movie_to_index[movie_id]
                    return movie_id, movie_index
        return None
    
    def compute_user_profile(self, 
                            user_ratings: List[Tuple[int, float]],
                            fit_new_user_func,
                            K: int,
                            lambda_reg: float,
                            tau_reg: float) -> Tuple[np.ndarray, float]:
        """
        Compute user factors and bias from ratings.
        
        Args:
            user_ratings: List of (movie_index, rating) tuples
            fit_new_user_func: Function to fit new user profile
            K: Number of latent factors
            lambda_reg: Regularization parameter
            tau_reg: Regularization parameter for bias
            
        Returns:
            Tuple of (user_factors, user_bias)
        """
        user_factor, user_bias = fit_new_user_func(
            user_ratings,
            self.item_factors,
            self.item_biases,
            K,
            lambda_reg,
            tau_reg
        )
        return user_factor, user_bias
    
    def generate_predictions(self,
                           user_factor: np.ndarray,
                           user_bias: float,
                           predict_func,
                           beta_values: List[float]) -> Dict[str, np.ndarray]:
        """
        Generate predictions using different beta values.
        
        Args:
            user_factor: User latent factors
            user_bias: User bias term
            predict_func: Prediction function
            beta_values: List of beta values to test
            
        Returns:
            Dictionary mapping method names to prediction arrays
        """
        scoring_methods = {}
        for beta in beta_values:
            method_name = f'Method with β={beta}'
            predictions = predict_func(
                user_factor,
                self.item_factors,
                user_bias,
                self.item_biases,
                beta
            )
            scoring_methods[method_name] = predictions
        
        return scoring_methods
    
    def plot_individual_recommendations(self,
                                       method_name: str,
                                       predictions: np.ndarray,
                                       target_movie_index: Optional[int] = None,
                                       top_n: int = 10):
        """
        Create a horizontal bar chart of top recommendations for a single method.
        
        Args:
            method_name: Name of the scoring method
            predictions: Array of prediction scores for all movies
            target_movie_index: Index of target movie (optional)
            top_n: Number of top recommendations to show
        """
        print(f"\n{'─'*80}")
        print(f"METHOD: {method_name}")
        print(f"{'─'*80}")
        
        # Sort movies by prediction score
        top_movie_indices = np.argsort(predictions)[::-1]
        
        # Find target movie rank if provided
        if target_movie_index is not None:
            target_rank = np.where(top_movie_indices == target_movie_index)[0][0] + 1
            target_score = predictions[target_movie_index]
            target_title = self.movie_id_to_title.get(
                self.index_to_movie[target_movie_index], 
                "Unknown"
            )
            print(f"'{target_title}' is ranked #{target_rank} with score {target_score:.4f}")
        
        # Collect top N movies
        movies = []
        scores = []
        
        for i in top_movie_indices[:top_n]:
            movie_id = self.index_to_movie[i]
            title = self.movie_id_to_title.get(movie_id, "Unknown title")
            score = predictions[i]
            
            display_title = self.wrap_title(title, max_width=50)
            movies.append(display_title)
            scores.append(score)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        base_color = self.method_colors.get(method_name, '#3498DB')
        
        # Create horizontal bars
        y_pos = np.arange(len(movies))
        bars = ax.barh(y_pos, scores,
                      color=base_color,
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=1.5)
        
        # Add gradient effect
        for bar, score in zip(bars, scores):
            bar.set_alpha(0.5 + 0.5 * (score / max(scores)))
        
        # Add score labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}',
                   ha='left', va='center',
                   fontweight='bold',
                   fontsize=10,
                   color='#2C3E50')
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(movies)
        ax.invert_yaxis()
        ax.set_xlabel('Prediction Score', fontweight='bold', fontsize=12)
        ax.set_title(f'Top {top_n} Movie Recommendations\nMethod: {method_name}',
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save
        safe_name = method_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        pdf_path = os.path.join(self.output_dir, f'01_recommendations_{safe_name}.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        
        plt.show()
        plt.close()
        
        # Print text summary
        print(f"\nTop {top_n} recommendations:")
        for rank, (movie, score) in enumerate(zip(movies, scores), 1):
            movie_single_line = movie.replace('\n', ' ')
            print(f"  {rank:2d}. {movie_single_line:<50} | Score: {score:.4f}")
    
    def plot_method_comparison(self,
                              scoring_methods: Dict[str, np.ndarray],
                              top_n: int = 10):
        """
        Create side-by-side comparison of multiple methods.
        
        Args:
            scoring_methods: Dictionary mapping method names to prediction arrays
            top_n: Number of top recommendations to show
        """
        print(f"\n{'='*80}")
        print("GENERATING COMPARISON VISUALIZATION")
        print(f"{'='*80}")
        
        # Prepare data for each method
        comparison_data = {}
        for method_name, predictions in scoring_methods.items():
            top_indices = np.argsort(predictions)[::-1][:top_n]
            
            movies = []
            scores = []
            for i in top_indices:
                movie_id = self.index_to_movie[i]
                title = self.movie_id_to_title.get(movie_id, "Unknown title")
                display_title = self.wrap_title(title, max_width=35)
                movies.append(display_title)
                scores.append(predictions[i])
            
            comparison_data[method_name] = {
                'movies': movies,
                'scores': scores
            }
        
        # Create subplots
        n_methods = len(scoring_methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(7*n_methods, 9), sharey=True)
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle(f'Comparison of Recommendation Methods - Top {top_n} Movies',
                    fontweight='bold', fontsize=16, y=0.98)
        
        # Plot each method
        for idx, (method_name, data) in enumerate(comparison_data.items()):
            ax = axes[idx]
            movies = data['movies']
            scores = data['scores']
            
            base_color = self.method_colors.get(method_name, '#3498DB')
            
            y_pos = np.arange(len(movies))
            bars = ax.barh(y_pos, scores,
                          color=base_color,
                          alpha=0.75,
                          edgecolor='black',
                          linewidth=1.2)
            
            # Add gradient
            for bar, score in zip(bars, scores):
                bar.set_alpha(0.5 + 0.5 * (score / max(scores)))
            
            # Add score labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.2f}',
                       ha='left', va='center',
                       fontsize=8,
                       fontweight='bold',
                       color='#2C3E50')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(movies, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_title(method_name, fontweight='bold', fontsize=12, pad=10)
            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Sync x-axis
            if idx > 0:
                ax.set_xlim(axes[0].get_xlim())
        
        plt.tight_layout()
        
        pdf_path = os.path.join(self.output_dir, '02_comparison_all_methods.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        
        plt.show()
        plt.close()
    
    def plot_score_distribution(self,
                               method_name: str,
                               predictions: np.ndarray):
        """
        Plot the distribution of prediction scores with statistical analysis.
        
        Args:
            method_name: Name of the scoring method
            predictions: Array of prediction scores for all movies
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        base_color = self.method_colors.get(method_name, '#3498DB')
        
        # Create histogram
        n, bins, patches = ax.hist(predictions, bins=60,
                                   color=base_color,
                                   alpha=0.7,
                                   edgecolor='black',
                                   linewidth=0.8)
        
        # Calculate statistics
        mean_val = np.mean(predictions)
        median_val = np.median(predictions)
        std_val = np.std(predictions)
        
        # Normality test
        sample_size = min(5000, len(predictions))
        sample = np.random.choice(predictions, sample_size, replace=False)
        statistic, p_value = stats.shapiro(sample)
        
        # Customize plot
        ax.set_xlabel('Prediction Score', fontweight='bold', fontsize=13)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=13)
        ax.set_title(f'Score Distribution - {method_name}',
                    fontweight='bold', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save
        safe_name = method_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        pdf_path = os.path.join(self.output_dir, f'03_distribution_{safe_name}.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        
        # Print statistics
        print(f"\n{method_name}:")
        print(f"  Shapiro-Wilk W-statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4e}")
        if p_value > 0.05:
            print(f"  → Distribution appears to be NORMAL (p > 0.05)")
        else:
            print(f"  → Distribution deviates from NORMAL (p < 0.05)")
        print(f"  Skewness: {stats.skew(predictions):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(predictions):.4f}")
        
        plt.show()
        plt.close()
    
    def plot_statistics_summary(self, scoring_methods: Dict[str, np.ndarray]):
        """
        Create grouped bar chart comparing statistics across methods.
        
        Args:
            scoring_methods: Dictionary mapping method names to prediction arrays
        """
        print(f"\n{'='*80}")
        print("GENERATING STATISTICS SUMMARY")
        print(f"{'='*80}")
        
        # Compute statistics
        stats_data = []
        for method_name, predictions in scoring_methods.items():
            stats_data.append({
                'Method': method_name,
                'Mean': np.mean(predictions),
                'Median': np.median(predictions),
                'Std Dev': np.std(predictions),
                'Min': np.min(predictions),
                'Max': np.max(predictions)
            })
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Mean', 'Median', 'Std Dev', 'Max']
        x = np.arange(len(scoring_methods))
        width = 0.2
        colors = ['#3498DB', '#E74C3C', '#F39C12', '#9B59B6']
        
        for i, metric in enumerate(metrics):
            values = [s[metric] for s in stats_data]
            bars = ax.bar(x + i * width, values, width,
                         label=metric,
                         color=colors[i],
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=1.2)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Method', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title('Statistical Comparison of Scoring Methods',
                    fontweight='bold', fontsize=14, pad=15)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s['Method'] for s in stats_data], fontsize=10)
        ax.legend(title='Metric', loc='upper left', fontsize=10, title_fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        pdf_path = os.path.join(self.output_dir, '04_statistics_summary.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        
        plt.show()
        plt.close()
    
    def generate_full_report(self,
                            target_movie_title: str,
                            fit_new_user_func,
                            predict_func,
                            K: int,
                            lambda_reg: float,
                            tau_reg: float,
                            beta_values: List[float] = [0.05, 1],
                            top_n: int = 10):
        """
        Generate complete visualization report for a target movie.
        
        Args:
            target_movie_title: Title of movie to use as seed
            fit_new_user_func: Function to fit new user profile
            predict_func: Function to generate predictions
            K: Number of latent factors
            lambda_reg: Regularization parameter
            tau_reg: Regularization parameter for bias
            beta_values: List of beta values to test
            top_n: Number of top recommendations to show
        """
        print("\n" + "="*80)
        print("MOVIE RECOMMENDATIONS VISUALIZATION FOR REPORT")
        print("="*80)
        
        # Find target movie
        result = self.find_movie(target_movie_title)
        if result is None:
            print(f"Movie not found: '{target_movie_title}'")
            return
        
        target_movie_id, target_movie_index = result
        print(f"\nDummy user rates '{target_movie_title}' with 5 stars")
        
        # Compute user profile
        user_ratings = [(target_movie_index, 5.0)]
        user_factor, user_bias = self.compute_user_profile(
            user_ratings, fit_new_user_func, K, lambda_reg, tau_reg
        )
        print(f"User factors computed. User bias = {user_bias:.4f}")
        
        # Generate predictions
        scoring_methods = self.generate_predictions(
            user_factor, user_bias, predict_func, beta_values
        )
        
        # Generate visualizations
        print(f"\n{'='*80}")
        print("GENERATING INDIVIDUAL METHOD VISUALIZATIONS")
        print(f"{'='*80}")
        
        for method_name, predictions in scoring_methods.items():
            self.plot_individual_recommendations(
                method_name, predictions, target_movie_index, top_n
            )
        
        self.plot_method_comparison(scoring_methods, top_n)
        
        print(f"\n{'='*80}")
        print("GENERATING SCORE DISTRIBUTION VISUALIZATIONS")
        print(f"{'='*80}")
        
        for method_name, predictions in scoring_methods.items():
            self.plot_score_distribution(method_name, predictions)
        
        self.plot_statistics_summary(scoring_methods)
        
        print(f"\n{'='*80}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*80}")
        print(f"\nAll plots saved to: {self.output_dir}/")
        print("\nThese PDFs are ready for inclusion in your report!")
        print("They can be imported in LaTeX with: \\includegraphics{file.pdf}")

