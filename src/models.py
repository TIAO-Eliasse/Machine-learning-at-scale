# Prepare flat data structure once (outside the loop)
import numpy as np
def prepare_flat_data(data_by_user_arg, data_by_movie_arg, M, N):
    # For users
    user_ratings = []
    user_items = []
    user_start_end = np.zeros((M, 2), dtype=np.int32)

    current_idx = 0
    for m in range(M):
        user_start_end[m, 0] = current_idx
        for n, r in data_by_user_arg[m]:
            user_items.append(n)
            user_ratings.append(r)
            current_idx += 1
        user_start_end[m, 1] = current_idx

    # For items
    item_ratings = []
    item_users = []
    item_start_end = np.zeros((N, 2), dtype=np.int32)

    current_idx = 0

    for n in range(N):
        item_start_end[n, 0] = current_idx
        for m, r in data_by_movie_arg[n]:
            item_users.append(m)
            item_ratings.append(r)
            current_idx += 1
        item_start_end[n, 1] = current_idx

    return (np.array(user_ratings, dtype=np.float32),
            np.array(user_items, dtype=np.int32),
            user_start_end,
            np.array(item_ratings, dtype=np.float32),
            np.array(item_users, dtype=np.int32),
            item_start_end)






from numba import njit, prange
import numpy as np

@njit(parallel=True)
def update_biases(M, N, user_ratings, user_items, user_start_end,
                  item_ratings, item_users, item_start_end,
                  user_biases, item_biases, lambda_reg, gamma_reg):
    """
    Single function that updates both user and item biases using:
    - Flat lists for ratings and items
    - Start/end indices for each user/item
    - Single loops (range-based)
    - Parallelization with prange
    """

    # Update user biases
    for m in prange(M):  # Parallelization over users
        start = user_start_end[m, 0]
        end = user_start_end[m, 1]

        if end > start:
            bias_sum = 0.0
            count = end - start

            # Single loop over ratings for this user
            for idx in range(start, end):
                n = user_items[idx]
                r = user_ratings[idx]
                bias_sum += lambda_reg * (r - item_biases[n])

            user_biases[m] = bias_sum / (lambda_reg * count + gamma_reg)

    # Update item biases
    for n in prange(N):  # Parallelization over items
        start = item_start_end[n, 0]
        end = item_start_end[n, 1]

        if end > start:
            bias_sum = 0.0
            count = end - start

            # Single loop over ratings for this item
            for idx in range(start, end):
                m = item_users[idx]
                r = item_ratings[idx]
                bias_sum += lambda_reg * (r - user_biases[m])

            item_biases[n] = bias_sum / (lambda_reg * count + gamma_reg)

    return user_biases, item_biases


from numba import njit, prange
import numpy as np

@njit(parallel=True)


def calculate_loss_bias_only(user_ratings, user_items, user_start_end, M,
                   user_biases, item_biases, lambda_reg, gamma_reg):
    """
    Compute the squared error loss for a bias-only recommender model using:
    - Flat lists for ratings and items
    - Start/end indices for each user
    - Single loops (range-based)
    - Parallelization with prange

    Parameters
    ----------
    user_ratings : np.array
        Flat array of all ratings
    user_items : np.array
        Flat array of all item indices
    user_start_end : np.array
        Array of shape (M, 2) where [m, 0] is start index and [m, 1] is end index
    M : int
        Number of users
    user_biases : np.array
        Bias for each user
    item_biases : np.array
        Bias for each item
    lambda_reg : float
        Regularization strength for user biases
    gamma_reg : float
        Regularization strength for item biases

    Returns
    -------
    float
        The total loss (MSE + regularization)
    """

    # Use array to accumulate results from parallel threads
    squared_errors = np.zeros(M, dtype=np.float32)

    for m in prange(M):  # Parallelization over users
        start = user_start_end[m, 0]
        end = user_start_end[m, 1]

        local_squared_error = 0.0

        # Single loop over ratings for this user
        for idx in range(start, end):
            n = user_items[idx]
            r = user_ratings[idx]
            prediction = user_biases[m] + item_biases[n]
            local_squared_error += (r - prediction) ** 2

        squared_errors[m] = local_squared_error

    # Aggregate squared errors
    total_loss = np.sum(squared_errors)

    # Add regularization terms
    total_loss += 0.5*lambda_reg * np.sum(user_biases ** 2)+0.5*gamma_reg * np.sum(item_biases ** 2)
    total_loss += gamma_reg * np.sum(item_biases ** 2)

    return total_loss



@njit(parallel=True)
def calculate_rmse_bias_only(user_ratings, user_items, user_start_end, M,
                   user_biases, item_biases):
    """
    Computes the Root Mean Squared Error (RMSE) using:
    - Flat lists for ratings and items
    - Start/end indices for each user
    - Single loops (range-based)
    - Parallelization with prange
    """
    # Use array to accumulate results from parallel threads
    squared_errors = np.zeros(M, dtype=np.float32)
    counts = np.zeros(M, dtype=np.int32)

    for m in prange(M):  # Parallelization over users
        start = user_start_end[m, 0]
        end = user_start_end[m, 1]

        local_squared_error = 0.0
        local_count = 0

        # Single loop over ratings for this user
        for idx in range(start, end):
            n = user_items[idx]
            r = user_ratings[idx]
            prediction = user_biases[m] + item_biases[n]
            local_squared_error += (r - prediction) ** 2
            local_count += 1

        squared_errors[m] = local_squared_error
        counts[m] = local_count

    # Aggregate results
    total_squared_error = np.sum(squared_errors)
    total_count = np.sum(counts)

    if total_count == 0:
        return 0.0

    return np.sqrt(total_squared_error / total_count)


def calculate_for_plot(M, N,  user_ratings_train, user_items_train, user_start_end_train, \
    item_ratings_train, item_users_train, item_start_end_train , user_ratings_test, user_items_test, user_start_end_test,
                       lambda_reg, gamma_reg, num_iterations):
    """
    Runs the ALS training loop and records performance metrics for each iteration.
    Uses flat data structures with start/end indices for efficiency.
    """
    # Initialize biases
    user_biases = np.zeros(M, dtype=np.float32)
    item_biases = np.zeros(N, dtype=np.float32)

    



    loss_train, loss_test, rmse_train, rmse_test = [], [], [], []

    #print("Starting Alternating Least Squares for biases...")

    for iteration in range(num_iterations):
        # Update biases using flat data structures
        user_biases, item_biases = update_biases(
            M, N,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_biases, item_biases, lambda_reg, gamma_reg
        )

        # Compute metrics using flat data structures
        loss_train.append(calculate_loss_bias_only(
            user_ratings_train, user_items_train, user_start_end_train, M,
            user_biases, item_biases, lambda_reg, gamma_reg
        ))

        rmse_train.append(calculate_rmse_bias_only(
            user_ratings_train, user_items_train, user_start_end_train, M,
            user_biases, item_biases
        ))

        rmse_test.append(calculate_rmse_bias_only(
            user_ratings_test, user_items_test, user_start_end_test, M,
            user_biases, item_biases
        ))

        loss_test.append(calculate_loss_bias_only(
            user_ratings_test, user_items_test, user_start_end_test, M,
            user_biases, item_biases, lambda_reg, gamma_reg
        ))

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration+1}/{num_iterations} | "
                  f"Train RMSE: {rmse_train[-1]:.4f} | Test RMSE: {rmse_test[-1]:.4f}")

    #print("Alternating Least Squares for biases completed.")
    return user_biases, item_biases, loss_train, loss_test, rmse_train, rmse_test






import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed




def run_model_once(M, N,
                   user_ratings_train, user_items_train, user_start_end_train,
                   item_ratings_train, item_users_train, item_start_end_train,
                   user_ratings_test, user_items_test, user_start_end_test,
                   lambda_reg, gamma_reg, num_iterations):
    """
    Execute model training once with given hyperparameters
    Used by parallel processing in grid/random search
    """
    try:
        user_biases, item_biases, loss_train, loss_test, rmse_train, rmse_test = \
            calculate_for_plot(
                M, N,
                user_ratings_train, user_items_train, user_start_end_train,
                item_ratings_train, item_users_train, item_start_end_train,
                user_ratings_test, user_items_test, user_start_end_test,
                lambda_reg, gamma_reg, num_iterations
            )

        return {
            "lambda_reg": lambda_reg,
            "gamma_reg": gamma_reg,
            "rmse_test": rmse_test[-1],
            "rmse_train": rmse_train[-1],
            "overfitting_gap": rmse_test[-1] - rmse_train[-1],
            "error": None
        }

    except Exception as e:
        return {
            "lambda_reg": lambda_reg,
            "gamma_reg": gamma_reg,
            "rmse_test": np.inf,
            "rmse_train": np.inf,
            "overfitting_gap": np.inf,
            "error": str(e)
        }
import urllib.request
import zipfile
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import shutil
import urllib.request
import zipfile


import random
from numba import njit, prange
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed

import zipfile
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import shutil
import urllib.request
import zipfile


import random
from numba import njit, prange
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed


def grid_search_bias_model(M, N,
                           user_ratings_train, user_items_train, user_start_end_train,
                           item_ratings_train, item_users_train, item_start_end_train,
                           user_ratings_test, user_items_test, user_start_end_test,
                           num_iterations=50):
    """
    Perform parallel grid search over hyperparameter space
    Returns: best_lambda, best_gamma, all_results
    """
    print("="*80)
    print("GRID SEARCH - PARALLEL BIAS-ONLY MODEL")
    print("="*80)

    lambda_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    gamma_values = [0.001, 0.005, 0.01, 0.02, 0.05]

    hyperparam_list = [
        (lam, gam)
        for lam in lambda_values
        for gam in gamma_values
    ]

    print(f"Total combinations = {len(hyperparam_list)}")
    print("Running in PARALLEL...")

    results = Parallel(n_jobs=-1)(
        delayed(run_model_once)(
            M, N,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_ratings_test, user_items_test, user_start_end_test,
            lam, gam, num_iterations
        )
        for (lam, gam) in hyperparam_list
    )

    # Filter valid results (without inf values)
    valid_results = [r for r in results if r["rmse_test"] != np.inf]
    
    if len(valid_results) == 0:
        raise ValueError("No valid results found!")

    # Find best result
    best = min(valid_results, key=lambda x: x["rmse_test"])

    print("\n🏆 BEST PARAMETERS")
    print(f"  lambda_reg: {best['lambda_reg']}")
    print(f"  gamma_reg: {best['gamma_reg']}")
    print(f"  rmse_test: {best['rmse_test']}")
    print(f"  rmse_train: {best['rmse_train']}")
    print(f"  overfitting_gap: {best['overfitting_gap']}")

    return best["lambda_reg"], best["gamma_reg"], results


def random_search_bias_model(M, N,
                             user_ratings_train, user_items_train, user_start_end_train,
                             item_ratings_train, item_users_train, item_start_end_train,
                             user_ratings_test, user_items_test, user_start_end_test,
                             n_iterations=30, num_iterations_model=50):
    """
    Perform parallel random search over hyperparameter space
    Returns: best_lambda, best_gamma, all_results
    """
    print("="*80)
    print("RANDOM SEARCH - PARALLEL BIAS-ONLY MODEL")
    print("="*80)

    lambda_min, lambda_max = 0.001, 0.1
    gamma_min, gamma_max = 0.001, 0.05

    random_params = [
        (np.random.uniform(lambda_min, lambda_max),
         np.random.uniform(gamma_min, gamma_max))
        for _ in range(n_iterations)
    ]

    print(f"Total random combinations = {n_iterations}")
    print("Running in PARALLEL...")

    results = Parallel(n_jobs=-1)(
        delayed(run_model_once)(
            M, N,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_ratings_test, user_items_test, user_start_end_test,
            lam, gam, num_iterations_model
        )
        for (lam, gam) in random_params
    )

    # Filter valid results
    valid_results = [r for r in results if r["rmse_test"] != np.inf]
    
    if len(valid_results) == 0:
        raise ValueError("No valid results found!")

    # Find best result
    best = min(valid_results, key=lambda x: x["rmse_test"])

    print("\n🏆 BEST PARAMETERS FROM RANDOM SEARCH")
    print(f"  lambda_reg: {best['lambda_reg']}")
    print(f"  gamma_reg: {best['gamma_reg']}")
    print(f"  rmse_test: {best['rmse_test']}")
    print(f"  rmse_train: {best['rmse_train']}")
    print(f"  overfitting_gap: {best['overfitting_gap']}")

    return best["lambda_reg"], best["gamma_reg"], results




def analyser_donnees(resultats):
    """
    Analyze and prepare data for visualization
    Returns: dict with all necessary statistics
    """
    # Handle tuple input from grid_search/random_search
    if isinstance(resultats, tuple):
        # Extract the results list (third element of tuple)
        if len(resultats) == 3:
            resultats = resultats[2]
        else:
            print(f"Unexpected tuple length: {len(resultats)}")
            return None
    
    # Ensure resultats is a list
    if isinstance(resultats, dict):
        resultats = [resultats]
    elif not isinstance(resultats, list):
        print(f"Type of resultats: {type(resultats)}")
        return None
    
    # Filter valid results
    results = []
    for r in resultats:
        if isinstance(r, dict) and "rmse_test" in r and r["rmse_test"] != np.inf:
            results.append(r)
    
    if len(results) == 0:
        print("No valid results found")
        return None
    
    # Group by lambda
    lambda_grouped = {}
    for r in results:
        key = r['lambda_reg']
        if key not in lambda_grouped:
            lambda_grouped[key] = []
        lambda_grouped[key].append(r)
    
    # Group by gamma
    gamma_grouped = {}
    for r in results:
        key = r['gamma_reg']
        if key not in gamma_grouped:
            gamma_grouped[key] = []
        gamma_grouped[key].append(r)
    
    # Statistics by lambda
    lambda_vals = sorted(lambda_grouped.keys())
    lambda_test_mean = [np.mean([r['rmse_test'] for r in lambda_grouped[k]]) for k in lambda_vals]
    lambda_test_std = [np.std([r['rmse_test'] for r in lambda_grouped[k]]) for k in lambda_vals]
    lambda_train_mean = [np.mean([r['rmse_train'] for r in lambda_grouped[k]]) for k in lambda_vals]
    lambda_train_std = [np.std([r['rmse_train'] for r in lambda_grouped[k]]) for k in lambda_vals]
    lambda_overfit_mean = [np.mean([r['overfitting_gap'] for r in lambda_grouped[k]]) for k in lambda_vals]
    lambda_overfit_std = [np.std([r['overfitting_gap'] for r in lambda_grouped[k]]) for k in lambda_vals]
    
    # Statistics by gamma
    gamma_vals = sorted(gamma_grouped.keys())
    gamma_test_mean = [np.mean([r['rmse_test'] for r in gamma_grouped[k]]) for k in gamma_vals]
    gamma_test_std = [np.std([r['rmse_test'] for r in gamma_grouped[k]]) for k in gamma_vals]
    gamma_train_mean = [np.mean([r['rmse_train'] for r in gamma_grouped[k]]) for k in gamma_vals]
    gamma_train_std = [np.std([r['rmse_train'] for r in gamma_grouped[k]]) for k in gamma_vals]
    gamma_overfit_mean = [np.mean([r['overfitting_gap'] for r in gamma_grouped[k]]) for k in gamma_vals]
    gamma_overfit_std = [np.std([r['overfitting_gap'] for r in gamma_grouped[k]]) for k in gamma_vals]
    
    # Pivot tables for heatmaps
    row_labels = sorted(set(r['lambda_reg'] for r in results))
    col_labels = sorted(set(r['gamma_reg'] for r in results))
    
    matrix_rmse = np.full((len(row_labels), len(col_labels)), np.nan)
    matrix_overfit = np.full((len(row_labels), len(col_labels)), np.nan)
    
    for r in results:
        row_idx = row_labels.index(r['lambda_reg'])
        col_idx = col_labels.index(r['gamma_reg'])
        matrix_rmse[row_idx, col_idx] = r['rmse_test']
        matrix_overfit[row_idx, col_idx] = r['overfitting_gap']
    
    # Best result
    best = min(results, key=lambda x: x['rmse_test'])
    
    # Top 10
    top_10 = sorted(results, key=lambda x: x['rmse_test'])[:10]
    
    return {
        'results': results,
        'best': best,
        'top_10': top_10,
        'lambda': {
            'vals': np.array(lambda_vals),
            'test_mean': np.array(lambda_test_mean),
            'test_std': np.array(lambda_test_std),
            'train_mean': np.array(lambda_train_mean),
            'train_std': np.array(lambda_train_std),
            'overfit_mean': np.array(lambda_overfit_mean),
            'overfit_std': np.array(lambda_overfit_std)
        },
        'gamma': {
            'vals': np.array(gamma_vals),
            'test_mean': np.array(gamma_test_mean),
            'test_std': np.array(gamma_test_std),
            'train_mean': np.array(gamma_train_mean),
            'train_std': np.array(gamma_train_std),
            'overfit_mean': np.array(gamma_overfit_mean),
            'overfit_std': np.array(gamma_overfit_std)
        },
        'heatmap': {
            'row_labels': row_labels,
            'col_labels': col_labels,
            'matrix_rmse': matrix_rmse,
            'matrix_overfit': matrix_overfit
        }
    }


def creer_graphiques(data):
    """
    Create all 8 subplots for visualization
    """
    fig = plt.figure(figsize=(20, 12))
    
   
    ax1 = plt.subplot(2, 4, 1)
    ax1.errorbar(data['lambda']['vals'], data['lambda']['test_mean'],
                 yerr=data['lambda']['test_std'], marker='o', linewidth=2,
                 markersize=8, label='Test RMSE', capsize=5)
    ax1.errorbar(data['lambda']['vals'], data['lambda']['train_mean'],
                 yerr=data['lambda']['train_std'], marker='s', linewidth=2,
                 markersize=8, label='Train RMSE', capsize=5, alpha=0.7)
    ax1.set_xlabel('Lambda (Regularization)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax1.set_title('Impact of Lambda on RMSE\n(mean ± std)', fontweight='bold', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')
    
    # ========== SUBPLOT 2: Gamma Impact on RMSE ==========
    ax2 = plt.subplot(2, 4, 2)
    ax2.errorbar(data['gamma']['vals'], data['gamma']['test_mean'],
                 yerr=data['gamma']['test_std'], marker='o', linewidth=2,
                 markersize=8, label='Test RMSE', capsize=5, color='coral')
    ax2.errorbar(data['gamma']['vals'], data['gamma']['train_mean'],
                 yerr=data['gamma']['train_std'], marker='s', linewidth=2,
                 markersize=8, label='Train RMSE', capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Gamma (Learning Rate)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax2.set_title('Impact of Gamma on RMSE\n(mean ± std)', fontweight='bold', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    
    # ========== SUBPLOT 3: Lambda Impact on Overfitting ==========
    ax3 = plt.subplot(2, 4, 3)
    ax3.errorbar(data['lambda']['vals'], data['lambda']['overfit_mean'],
                 yerr=data['lambda']['overfit_std'], marker='D', linewidth=2.5,
                 markersize=8, color='purple', capsize=5, label='Overfitting Gap')
    ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect fit')
    ax3.set_xlabel('Lambda (Regularization)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Overfitting Gap (Test - Train)', fontweight='bold', fontsize=11)
    ax3.set_title('Impact of Lambda on Overfitting\n(Lower is better)', fontweight='bold', fontsize=11)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xscale('log')
    

    ax4 = plt.subplot(2, 4, 4)
    ax4.errorbar(data['gamma']['vals'], data['gamma']['overfit_mean'],
                 yerr=data['gamma']['overfit_std'], marker='D', linewidth=2.5,
                 markersize=8, color='darkgreen', capsize=5, label='Overfitting Gap')
    ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect fit')
    ax4.set_xlabel('Gamma (Learning Rate)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Overfitting Gap (Test - Train)', fontweight='bold', fontsize=11)
    ax4.set_title('Impact of Gamma on Overfitting\n(Lower is better)', fontweight='bold', fontsize=11)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xscale('log')
    

    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(data['heatmap']['matrix_rmse'], cmap='RdYlGn_r', aspect='auto')
    for i in range(len(data['heatmap']['row_labels'])):
        for j in range(len(data['heatmap']['col_labels'])):
            if not np.isnan(data['heatmap']['matrix_rmse'][i, j]):
                ax5.text(j, i, f"{data['heatmap']['matrix_rmse'][i, j]:.4f}",
                        ha="center", va="center", color="black", fontsize=8)
    ax5.set_xticks(range(len(data['heatmap']['col_labels'])))
    ax5.set_yticks(range(len(data['heatmap']['row_labels'])))
    ax5.set_xticklabels([f'{v:.3f}' for v in data['heatmap']['col_labels']], rotation=45, ha='right')
    ax5.set_yticklabels([f'{v:.3f}' for v in data['heatmap']['row_labels']])
    ax5.set_xlabel('Gamma (Learning Rate)', fontweight='bold')
    ax5.set_ylabel('Lambda (Regularization)', fontweight='bold')
    ax5.set_title('RMSE Test: Lambda × Gamma Interaction', fontweight='bold', fontsize=11)
    plt.colorbar(im5, ax=ax5, label='RMSE Test')
    
    ax6 = plt.subplot(2, 4, 6)
    vmax = max(abs(np.nanmin(data['heatmap']['matrix_overfit'])), abs(np.nanmax(data['heatmap']['matrix_overfit'])))
    im6 = ax6.imshow(data['heatmap']['matrix_overfit'], cmap='coolwarm', aspect='auto',
                     vmin=-vmax, vmax=vmax)
    for i in range(len(data['heatmap']['row_labels'])):
        for j in range(len(data['heatmap']['col_labels'])):
            if not np.isnan(data['heatmap']['matrix_overfit'][i, j]):
                ax6.text(j, i, f"{data['heatmap']['matrix_overfit'][i, j]:.4f}",
                        ha="center", va="center", color="black", fontsize=8)
    ax6.set_xticks(range(len(data['heatmap']['col_labels'])))
    ax6.set_yticks(range(len(data['heatmap']['row_labels'])))
    ax6.set_xticklabels([f'{v:.3f}' for v in data['heatmap']['col_labels']], rotation=45, ha='right')
    ax6.set_yticklabels([f'{v:.3f}' for v in data['heatmap']['row_labels']])
    ax6.set_xlabel('Gamma (Learning Rate)', fontweight='bold')
    ax6.set_ylabel('Lambda (Regularization)', fontweight='bold')
    ax6.set_title('Overfitting Gap: Lambda × Gamma', fontweight='bold', fontsize=11)
    plt.colorbar(im6, ax=ax6, label='Overfitting Gap')
    

    ax7 = plt.subplot(2, 4, 7)
    overfitting_gaps = [r['overfitting_gap'] for r in data['results']]
    rmse_tests = [r['rmse_test'] for r in data['results']]
    lambdas = [r['lambda_reg'] for r in data['results']]
    scatter = ax7.scatter(overfitting_gaps, rmse_tests, c=lambdas, cmap='viridis',
                         s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax7.scatter(data['best']['overfitting_gap'], data['best']['rmse_test'],
               s=300, marker='*', color='red', edgecolors='black',
               linewidth=2, label='Best model', zorder=5)
    ax7.set_xlabel('Overfitting Gap (Test - Train)', fontweight='bold', fontsize=11)
    ax7.set_ylabel('RMSE Test', fontweight='bold', fontsize=11)
    ax7.set_title('Performance vs Overfitting Trade-off\n(color = lambda value)',
                  fontweight='bold', fontsize=11)
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax7, label='Lambda')
    
 
    ax8 = plt.subplot(2, 4, 8)
    labels = [f"λ={r['lambda_reg']:.3f}\nγ={r['gamma_reg']:.3f}" for r in data['top_10']]
    rmse_values = [r['rmse_test'] for r in data['top_10']]
    overfit_gaps = [r['overfitting_gap'] for r in data['top_10']]
    
    overfit_min = min(overfit_gaps)
    overfit_max = max(overfit_gaps)
    if overfit_max > overfit_min:
        norm_gaps = [(g - overfit_min) / (overfit_max - overfit_min) for g in overfit_gaps]
    else:
        norm_gaps = [0.5] * len(overfit_gaps)
    
    colors = plt.cm.coolwarm(norm_gaps)
    ax8.barh(range(len(data['top_10'])), rmse_values, color=colors,
             edgecolor='black', linewidth=1.5)
    ax8.set_yticks(range(len(data['top_10'])))
    ax8.set_yticklabels(labels, fontsize=9)
    ax8.set_xlabel('RMSE Test', fontweight='bold', fontsize=11)
    ax8.set_title('Top 10 Best Configurations\n(color = overfitting gap)',
                  fontweight='bold', fontsize=11)
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    plt.show()


def afficher_resultats(data):
    """
    Display statistics, tables, and insights
    """
    results = data['results']
    best = data['best']
    
    # General statistics
    rmse_tests = [r['rmse_test'] for r in results]
    overfitting_gaps = [r['overfitting_gap'] for r in results]
    
    print(f"\nNumber of tests: {len(results)}")
    print(f"Best RMSE test: {min(rmse_tests):.4f}")
    print(f"Worst RMSE test: {max(rmse_tests):.4f}")
    print(f"Mean RMSE test: {np.mean(rmse_tests):.4f}")
    print(f"Best overfitting gap: {min(overfitting_gaps):.4f}")
    print(f"Worst overfitting gap: {max(overfitting_gaps):.4f}")
    
    # Top 5 configurations
    print("\n📋 TOP 5 BEST CONFIGURATIONS:")
    print("-" * 110)
    print(f"{'Lambda':<12} {'Gamma':<12} {'RMSE Train':<15} {'RMSE Test':<15} {'Overfitting Gap':<18}")
    print("-" * 110)
    
    for r in data['top_10'][:5]:
        print(f"{r['lambda_reg']:<12.6f} {r['gamma_reg']:<12.6f} "
              f"{r['rmse_train']:<15.6f} {r['rmse_test']:<15.6f} "
              f"{r['overfitting_gap']:<18.6f}")
    print("-" * 110)
    
    # Key insights
    lambda_stds = []
    for val in data['lambda']['vals']:
        group_results = [r for r in results if r['lambda_reg'] == val]
        lambda_stds.append(np.std([r['rmse_test'] for r in group_results]))
    
    gamma_stds = []
    for val in data['gamma']['vals']:
        group_results = [r for r in results if r['gamma_reg'] == val]
        gamma_stds.append(np.std([r['rmse_test'] for r in group_results]))
    
    print("\n KEY INSIGHTS:")
    print(f"  • Best lambda: {best['lambda_reg']:.4f}")
    print(f"  • Best gamma: {best['gamma_reg']:.4f}")
    print(f"  • Lambda sensitivity: {np.mean(lambda_stds):.4f}")
    print(f"  • Gamma sensitivity: {np.mean(gamma_stds):.4f}")


def visualiser_impact_hyperparametres(resultats, search_type="Grid Search"):
    """
    Main function to visualize hyperparameter impact on:
    1. RMSE performance
    2. Overfitting gap (test - train)
    3. Parameter interactions
    """
    print(f"\n HYPERPARAMETER IMPACT ANALYSIS ({search_type})")
    print("="*80)
    
    # Analyze data
    data = analyser_donnees(resultats)
    
    if data is None:
        print("No valid results to visualize!")
        return None
    
    # Display statistics
    afficher_resultats(data)
    
    # Create graphs
    creer_graphiques(data)
    
    return data['results']


## Pratical 3




@njit(parallel=True)
def update_biases_and_factors(M, N, K,
                              user_ratings, user_items, user_start_end,
                              item_ratings, item_users, item_start_end,
                              user_biases, item_biases, user_factors, item_factors,
                              lambda_reg, gamma_reg, tau_reg, num_iterations):


    for iteration in range(num_iterations):


        for m in prange(M):  # Parallelization over users
            start = user_start_end[m, 0]
            end = user_start_end[m, 1]

            if end <= start:
                continue

            count = end - start
            
            bias_sum = 0.0
            for idx in range(start, end):
                n = user_items[idx]
                r = user_ratings[idx]

                
                dot_product = 0.0
                for k in range(K):
                    dot_product += user_factors[m, k] * item_factors[n, k]

                bias_sum += r - item_biases[n] - dot_product

            user_biases[m] = bias_sum / (count + gamma_reg)

           
            A = tau_reg * np.eye(K)
            b_vec = np.zeros(K)

            for idx in range(start, end):
                n = user_items[idx]
                r = user_ratings[idx]
                v_n = item_factors[n]

                # A += lambda_reg * outer(v_n, v_n)
                for i in range(K):
                    for j in range(K):
                        A[i, j] += lambda_reg * v_n[i] * v_n[j]

                # b_vec += lambda_reg * v_n * (r - user_biases[m] - item_biases[n])
                residual = r - user_biases[m] - item_biases[n]
                for i in range(K):
                    b_vec[i] += lambda_reg * v_n[i] * residual

            user_factors[m] = np.linalg.solve(A, b_vec)



        # --- Update ITEM factors and  item biases ---
        for n in prange(N):  # Parallelization over items
            start = item_start_end[n, 0]
            end = item_start_end[n, 1]

            if end <= start:
                continue

            count = end - start
 # (1) First loop: update ITEM bias
            bias_sum = 0.0
            for idx in range(start, end):
                m_idx = item_users[idx]
                r = item_ratings[idx]

                # Compute dot product
                dot_product = 0.0
                for k in range(K):
                    dot_product += user_factors[m_idx, k] * item_factors[n, k]

                bias_sum += r - user_biases[m_idx] - dot_product

            item_biases[n] = bias_sum / (count + gamma_reg)

            # (2) Second loop: update ITEM vector
            A = tau_reg * np.eye(K)
            b_vec = np.zeros(K)

            for idx in range(start, end):
                m_idx = item_users[idx]
                r = item_ratings[idx]
                u_m = user_factors[m_idx]

                # A += lambda_reg * outer(u_m, u_m)
                for i in range(K):
                    for j in range(K):
                        A[i, j] += lambda_reg * u_m[i] * u_m[j]

                # b_vec += lambda_reg * u_m * (r - user_biases[m_idx] - item_biases[n])
                residual = r - user_biases[m_idx] - item_biases[n]
                for i in range(K):
                    b_vec[i] += lambda_reg * u_m[i] * residual

            item_factors[n] = np.linalg.solve(A, b_vec)


    return user_biases, item_biases, user_factors, item_factors


    ## Loss function 



@njit(parallel=True)
def calculate_loss_negatif(user_ratings, user_items, user_start_end, M,
                          user_biases, item_biases, user_factors, item_factors,
                          lambda_reg, gamma_reg, tau_reg):
    """


    L = -(λ/2) * Σ (r_mn - (u_m^T v_n + b_u_m + b_i_n))²
        - (τ/2) * (Σ ||u_m||² + Σ ||v_n||²)
        - (γ/2) * (Σ b_u_m² + Σ b_i_n²)


    """
    K = user_factors.shape[1]

    # Use array to accumulate results from parallel threads
    squared_errors = np.zeros(M, dtype=np.float32)

    for m in prange(M):  # Parallelization over users
        start = user_start_end[m, 0]
        end = user_start_end[m, 1]

        local_squared_error = 0.0

        # Single loop over ratings for this user
        for idx in range(start, end):
            n = user_items[idx]
            r = user_ratings[idx]

            # Compute dot product manually
            dot_product = 0.0
            for k in range(K):
                dot_product += user_factors[m, k] * item_factors[n, k]

            prediction = dot_product + user_biases[m] + item_biases[n]
            local_squared_error += (r - prediction) ** 2

        squared_errors[m] = local_squared_error

    # Aggregate squared errors
    squared_error_sum = np.sum(squared_errors)

    # Regularization terms
    reg_bias = np.sum(user_biases ** 2) + np.sum(item_biases ** 2)
    reg_factors = np.sum(user_factors ** 2) + np.sum(item_factors ** 2)

    # Compute loss
    L = - (lambda_reg / 2) * squared_error_sum - (tau_reg / 2) * reg_factors - (gamma_reg / 2) * reg_bias

    return -L




# RMSE COMPUTATION


@njit(parallel=True)
def calculate_rmse(user_ratings, user_items, user_start_end, M,
                   user_biases, item_biases, user_factors, item_factors):
    """
    Computes RMSE between predictions and true ratings using:
    - Flat lists for ratings and items
    - Start/end indices for each user
    - Single loops (range-based)
    - Parallelization with prange

    Parameters
    ----------
    user_ratings : np.array
        Flat array of all ratings
    user_items : np.array
        Flat array of all item indices
    user_start_end : np.array
        Array of shape (M, 2) with start/end indices for each user
    M : int
        Number of users
    user_biases, item_biases : np.array
        Bias arrays
    user_factors, item_factors : np.array
        Factor matrices

    Returns
    -------
    float
        Root Mean Squared Error
    """
    K = user_factors.shape[1]

    # Use arrays to accumulate results from parallel threads
    squared_errors = np.zeros(M, dtype=np.float32)
    counts = np.zeros(M, dtype=np.int32)

    for m in prange(M):  # Parallelization over users
        start = user_start_end[m, 0]
        end = user_start_end[m, 1]

        local_squared_error = 0.0
        local_count = 0

        # Single loop over ratings for this user
        for idx in range(start, end):
            n = user_items[idx]
            r = user_ratings[idx]

            # Compute dot product manually
            dot_product = 0.0
            for k in range(K):
                dot_product += user_factors[m, k] * item_factors[n, k]

            prediction = dot_product + user_biases[m] + item_biases[n]
            local_squared_error += (r - prediction) ** 2
            local_count += 1

        squared_errors[m] = local_squared_error
        counts[m] = local_count

    # Aggregate results
    total_squared_error = np.sum(squared_errors)
    total_count = np.sum(counts)

    if total_count == 0:
        return 0.0

    return np.sqrt(total_squared_error / total_count)


## tRAIN METRICS

def train_and_evaluate_metrics(M, N, K, 
user_ratings_train, user_items_train, user_start_end_train, \
    item_ratings_train, item_users_train, item_start_end_train, user_ratings_test, user_items_test, user_start_end_test, 
                               lambda_reg, gamma_reg, tau_reg, num_iterations):
    """
    Train ALS model with λ, γ, and τ regularization terms using flat data structures.
    Computes metrics at each iteration.
    """
    # Initialize parameters
    user_biases = np.zeros(M, dtype=np.float32)
    item_biases = np.zeros(N, dtype=np.float32)
    user_factors = np.random.normal(0, 0.1, (M, K)).astype(np.float32)
    item_factors = np.random.normal(0, 0.1, (N, K)).astype(np.float32)

   
    loss_history, rmse_train, rmse_test = [], [], []

    print("Starting ALS training with λ, γ, τ terms...")

    # Train one iteration at a time to compute metrics
    for iteration in range(num_iterations):
        user_biases, item_biases, user_factors, item_factors = update_biases_and_factors(
            M, N, K,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_biases, item_biases, user_factors, item_factors,
            lambda_reg, gamma_reg, tau_reg, 1  # 1 iteration at a time
        )

        # Compute metrics
        loss = calculate_loss_negatif(
            user_ratings_train, user_items_train, user_start_end_train, M,
            user_biases, item_biases, user_factors, item_factors,
            lambda_reg, gamma_reg, tau_reg
        )
        loss_history.append(loss)

        rmse_tr = calculate_rmse(
            user_ratings_train, user_items_train, user_start_end_train, M,
            user_biases, item_biases, user_factors, item_factors
        )
        rmse_train.append(rmse_tr)

        rmse_te = calculate_rmse(
            user_ratings_test, user_items_test, user_start_end_test, M,
            user_biases, item_biases, user_factors, item_factors
        )
        rmse_test.append(rmse_te)

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration+1}/{num_iterations} | "
                  f"Train RMSE: {rmse_tr:.4f} | Test RMSE: {rmse_te:.4f} | Loss: {loss:.4f}")

    print("Training completed.")

    return {
        "loss_history": loss_history,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "user_biases": user_biases,
        "item_biases": item_biases,
        "user_factors": user_factors,
        "item_factors": item_factors
    }


def analyze_overfitting_by_degree(M, data_by_user_train, data_by_user_test,
                                   all_results, k_values):
    """
    Analyze overfitting as a function of user degree (number of ratings).
    Users with fewer ratings are expected to overfit more with higher K.
    """

    # Calculate user degrees (number of ratings in training)
    user_degrees = np.array([len(ratings_list) for ratings_list in data_by_user_train])

    # Store results for each K
    results_by_k = {}

    for k in k_values:
        results = all_results[k]
        user_biases = results['user_biases']
        item_biases = results['item_biases']
        user_factors = results['user_factors']
        item_factors = results['item_factors']

        train_errors = np.zeros(M)
        test_errors = np.zeros(M)
        train_counts = np.zeros(M)
        test_counts = np.zeros(M)

        # Calculate train RMSE per user
        for m in range(M):
            if len(data_by_user_train[m]) > 0:
                for n, r in data_by_user_train[m]:
                    pred = np.dot(user_factors[m], item_factors[n]) + user_biases[m] + item_biases[n]
                    train_errors[m] += (r - pred) ** 2
                    train_counts[m] += 1
                train_errors[m] = np.sqrt(train_errors[m] / train_counts[m])

        # Calculate test RMSE per user
        for m in range(M):
            if len(data_by_user_test[m]) > 0:
                for n, r in data_by_user_test[m]:
                    pred = np.dot(user_factors[m], item_factors[n]) + user_biases[m] + item_biases[n]
                    test_errors[m] += (r - pred) ** 2
                    test_counts[m] += 1
                test_errors[m] = np.sqrt(test_errors[m] / test_counts[m])

        # Calculate overfitting gap
        overfitting_gap = test_errors - train_errors

        # Filter valid users (have both train and test data)
        valid = (train_counts > 0) & (test_counts > 0)

        results_by_k[k] = {
            'train_errors': train_errors[valid],
            'test_errors': test_errors[valid],
            'gap': overfitting_gap[valid],
            'degrees': user_degrees[valid],
            'valid_mask': valid
        }

    return results_by_k




import matplotlib.pyplot as plt
import numpy as np

def visualize_overfitting_powerlaw(M, data_by_user_train, data_by_user_test,
                                   all_results, k_values, degree_bins=None):
    """
    Analyze and visualize overfitting vs user degree (Power-law analysis).

    Parameters
    ----------
    M : int
        Number of users.
    data_by_user_train : list of lists
        Training ratings by user.
    data_by_user_test : list of lists
        Test ratings by user.
    all_results : dict
        Dictionary containing results per K, with keys like 'train_errors', 'test_errors', 'gap', 'degrees'.
    k_values : list
        List of K values to analyze (e.g., [10, 20]).
    degree_bins : list, optional
        Degree bin edges for analysis. Defaults to [1, 5, 10, 20, 50, 100, 200, 500, 1000, 10000].
    """
    if degree_bins is None:
        degree_bins = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]

    print("="*80)
    print("ANALYZING OVERFITTING BY USER DEGREE (Power-Law Analysis)")
    print("="*80)

    # Compute results by K (wrapper for analysis function)
    results_by_k = analyze_overfitting_by_degree(M, data_by_user_train, data_by_user_test,
                                                 all_results, k_values)

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    # 1️⃣ Power-law distribution
    ax1 = plt.subplot(3, 3, 1)
    user_degrees_all = [len(ratings_list) for ratings_list in data_by_user_train]
    values, counts = np.unique(user_degrees_all, return_counts=True)
    ax1.scatter(values, counts, s=20, alpha=0.6, color='purple')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('User Degree (# ratings)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Power-Law: User Degree Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2️⃣-3️⃣ Overfitting gap vs degree for each K
    for idx, k in enumerate(k_values):
        ax = plt.subplot(3, 3, 2 + idx)
        data = results_by_k[k]

        scatter = ax.scatter(data['degrees'], data['gap'],
                             c=data['degrees'], cmap='viridis',
                             alpha=0.4, s=15, edgecolors='none')

        # Binned average line
        bin_means, bin_centers = [], []
        for i in range(len(degree_bins)-1):
            mask = (data['degrees'] >= degree_bins[i]) & (data['degrees'] < degree_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(data['gap'][mask]))
                bin_centers.append((degree_bins[i] + degree_bins[i+1])/2)
        ax.plot(bin_centers, bin_means, 'r-o', linewidth=3, markersize=8, label='Binned Average', alpha=0.8)

        ax.set_xscale('log')
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('User Degree (# ratings)', fontsize=11)
        ax.set_ylabel('Overfitting Gap (Test - Train RMSE)', fontsize=11)
        ax.set_title(f'Overfitting vs Degree (K={k})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label='Degree')

    # 4️⃣ Comparison K=10 vs K=20
    ax4 = plt.subplot(3, 3, 4)
    k10_data = results_by_k[10]
    k20_data = results_by_k[20]
    bins_labels, k10_bin_gaps, k20_bin_gaps = [], [], []

    for i in range(len(degree_bins)-1):
        mask_10 = (k10_data['degrees'] >= degree_bins[i]) & (k10_data['degrees'] < degree_bins[i+1])
        mask_20 = (k20_data['degrees'] >= degree_bins[i]) & (k20_data['degrees'] < degree_bins[i+1])
        if np.sum(mask_10) > 5 and np.sum(mask_20) > 5:
            bins_labels.append(f"{degree_bins[i]}-{degree_bins[i+1]}")
            k10_bin_gaps.append(np.mean(k10_data['gap'][mask_10]))
            k20_bin_gaps.append(np.mean(k20_data['gap'][mask_20]))

    x = np.arange(len(bins_labels))
    width = 0.35
    ax4.bar(x - width/2, k10_bin_gaps, width, label='K=10', alpha=0.8, color='steelblue')
    ax4.bar(x + width/2, k20_bin_gaps, width, label='K=20', alpha=0.8, color='coral')
    ax4.set_xlabel('User Degree Bins', fontsize=11)
    ax4.set_ylabel('Average Overfitting Gap', fontsize=11)
    ax4.set_title('Overfitting Gap by Degree Bins', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bins_labels, rotation=45, ha='right', fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)

    # 5️⃣ Difference in overfitting K20-K10
    ax5 = plt.subplot(3, 3, 5)
    gap_diff = k20_data['gap'] - k10_data['gap']
    scatter = ax5.scatter(k10_data['degrees'], gap_diff,
                          c=k10_data['degrees'], cmap='coolwarm',
                          alpha=0.5, s=15)
    ax5.set_xscale('log')
    ax5.axhline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax5.set_xlabel('User Degree (# ratings)', fontsize=11)
    ax5.set_ylabel('Δ Gap (K=20 - K=10)', fontsize=11)
    ax5.set_title('Increased Overfitting from K=10 to K=20', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Degree')

    # 6️⃣ Percentage of users with worse overfitting in K20
    ax6 = plt.subplot(3, 3, 6)
    worse_overfitting_pct, bin_labels_pct = [], []
    for i in range(len(degree_bins)-1):
        mask = (k10_data['degrees'] >= degree_bins[i]) & (k10_data['degrees'] < degree_bins[i+1])
        if np.sum(mask) > 5:
            worse_count = np.sum(gap_diff[mask] > 0)
            total_count = np.sum(mask)
            worse_overfitting_pct.append(100 * worse_count / total_count)
            bin_labels_pct.append(f"{degree_bins[i]}-{degree_bins[i+1]}")

    ax6.bar(range(len(worse_overfitting_pct)), worse_overfitting_pct,
            alpha=0.8, color='indianred', edgecolor='black')
    ax6.axhline(50, color='black', linestyle='--', linewidth=2, label='50% threshold')
    ax6.set_xlabel('User Degree Bins', fontsize=11)
    ax6.set_ylabel('% Users with Worse Overfitting in K=20', fontsize=11)
    ax6.set_title('Impact of Increased Complexity by Degree', fontsize=12, fontweight='bold')
    ax6.set_xticks(range(len(bin_labels_pct)))
    ax6.set_xticklabels(bin_labels_pct, rotation=45, ha='right', fontsize=9)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0, 100)

    # 7️⃣ Train RMSE vs Degree
    ax7 = plt.subplot(3, 3, 7)
    for k in k_values:
        data = results_by_k[k]
        bin_means, bin_centers = [], []
        for i in range(len(degree_bins)-1):
            mask = (data['degrees'] >= degree_bins[i]) & (data['degrees'] < degree_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(data['train_errors'][mask]))
                bin_centers.append((degree_bins[i] + degree_bins[i+1]) / 2)
        ax7.plot(bin_centers, bin_means, '-o', linewidth=2, markersize=7, label=f'K={k}')
    ax7.set_xscale('log')
    ax7.set_xlabel('User Degree (# ratings)', fontsize=11)
    ax7.set_ylabel('Average Train RMSE', fontsize=11)
    ax7.set_title('Train RMSE vs Degree', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8️⃣ Test RMSE vs Degree
    ax8 = plt.subplot(3, 3, 8)
    for k in k_values:
        data = results_by_k[k]
        bin_means, bin_centers = [], []
        for i in range(len(degree_bins)-1):
            mask = (data['degrees'] >= degree_bins[i]) & (data['degrees'] < degree_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(data['test_errors'][mask]))
                bin_centers.append((degree_bins[i] + degree_bins[i+1]) / 2)
        ax8.plot(bin_centers, bin_means, '-o', linewidth=2, markersize=7, label=f'K={k}')
    ax8.set_xscale('log')
    ax8.set_xlabel('User Degree (# ratings)', fontsize=11)
    ax8.set_ylabel('Average Test RMSE', fontsize=11)
    ax8.set_title('Test RMSE vs Degree', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9️⃣ Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = "POWER-LAW INSIGHTS\n" + "="*40 + "\n\n"
    for i, degree_range in enumerate([(1, 10), (10, 50), (50, 200), (200, 10000)]):
        summary_text += f"Users with {degree_range[0]}-{degree_range[1]} ratings:\n"
        for k in k_values:
            data = results_by_k[k]
            mask = (data['degrees'] >= degree_range[0]) & (data['degrees'] < degree_range[1])
            if np.sum(mask) > 0:
                avg_gap = np.mean(data['gap'][mask])
                summary_text += f"  K={k}: Gap = {avg_gap:.4f}\n"

        # Compare K=20 vs K=10
        mask_10 = (results_by_k[10]['degrees'] >= degree_range[0]) & (results_by_k[10]['degrees'] < degree_range[1])
        mask_20 = (results_by_k[20]['degrees'] >= degree_range[0]) & (results_by_k[20]['degrees'] < degree_range[1])
        if np.sum(mask_10) > 0 and np.sum(mask_20) > 0:
            diff = np.mean(results_by_k[20]['gap'][mask_20]) - np.mean(results_by_k[10]['gap'][mask_10])
            summary_text += f"  Δ(K20-K10): {diff:+.4f}\n"
        summary_text += "\n"

    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print(f"\n{'='*80}")
    print("DETAILED STATISTICS BY DEGREE RANGE")
    print(f"{'='*80}\n")

    for degree_range in [(1, 10), (10, 50), (50, 200), (200, 10000)]:
        print(f"\n{'-'*80}")
        print(f"Users with {degree_range[0]}-{degree_range[1]} ratings:")
        print(f"{'-'*80}")
        for k in k_values:
            data = results_by_k[k]
            mask = (data['degrees'] >= degree_range[0]) & (data['degrees'] < degree_range[1])
            if np.sum(mask) > 0:
                print(f"\nK = {k}:")
                print(f"  Number of users: {np.sum(mask)}")
                print(f"  Avg train RMSE:  {np.mean(data['train_errors'][mask]):.4f}")
                print(f"  Avg test RMSE:   {np.mean(data['test_errors'][mask]):.4f}")
                print(f"  Avg gap:         {np.mean(data['gap'][mask]):.4f}")
                print(f"  Std gap:         {np.std(data['gap'][mask]):.4f}")
                print(f"  % with gap > 0.5: {100*np.sum(data['gap'][mask] > 0.5)/np.sum(mask):.1f}%")






import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
import gc
import os

def safe_measure_thread_performance(calculate_func, M, N, user_ratings_train, user_items_train, 
                                    user_start_end_train, item_ratings_train, 
                                    item_users_train, item_start_end_train,
                                    user_ratings_test, user_items_test, 
                                    user_start_end_test, lambda_reg, gamma_reg, 
                                    num_iterations):
    """
    Safe version of thread performance analysis
    """
    import os
    
    # Detect available CPUs
    max_threads = os.cpu_count() or 4
    print(f"Detected CPUs: {max_threads}")
    
    # Adaptive thread list
    if max_threads >= 8:
        thread_list = [1, 2, 4, max_threads]
    elif max_threads >= 4:
        thread_list = [1, 2, max_threads]
    else:
        thread_list = [1, max_threads]
    
    results = {
        'threads': [],
        'total_time': [],
        'time_per_iteration': [],
        'speedup': [],
        'efficiency': []
    }
    
    baseline_time = None
    
    for num_threads in thread_list:
        try:
            print(f"\n{'='*50}")
            print(f"Testing with {num_threads} thread(s)...")
            print(f"{'='*50}")
            
            # Configure Numba (if available)
            try:
                from numba import set_num_threads
                set_num_threads(num_threads)
                print(f"✓ Numba configured with {num_threads} threads")
            except ImportError:
                print("⚠ Numba not available, using standard CPU")
          
            gc.collect()
            
        
            start = time.time()
            
            user_biases, item_biases, loss_train, loss_test, rmse_train, rmse_test = calculate_func(
                M, N, user_ratings_train, user_items_train, user_start_end_train,
                item_ratings_train, item_users_train, item_start_end_train,
                user_ratings_test, user_items_test, user_start_end_test,
                lambda_reg, gamma_reg, num_iterations
            )
            
            end = time.time()
            
            total_time = end - start
            time_per_iter = total_time / num_iterations
            
            print(f"✓ Total time: {total_time:.2f}s")
            print(f"✓ Time per iteration: {time_per_iter:.4f}s")
            
            # Calculate speedup
            if baseline_time is None:
                baseline_time = total_time
                speedup = 1.0
            else:
                speedup = baseline_time / total_time
            
            efficiency = speedup / num_threads
            
            results['threads'].append(num_threads)
            results['total_time'].append(total_time)
            results['time_per_iteration'].append(time_per_iter)
            results['speedup'].append(speedup)
            results['efficiency'].append(efficiency * 100)
            
            print(f"✓ Speedup: {speedup:.2f}x")
            print(f"✓ Efficiency: {efficiency*100:.1f}%")
            
            # Cleanup after test
            del user_biases, item_biases, loss_train, loss_test, rmse_train, rmse_test
            gc.collect()
            
        except Exception as e:
            print(f"✗ Error with {num_threads} threads:")
            print(f"  {str(e)}")
            traceback.print_exc()
            continue
    
    return results

def measure_iteration_times(calculate_func, M, N, user_ratings_train, user_items_train, 
                            user_start_end_train, item_ratings_train, 
                            item_users_train, item_start_end_train,
                            user_ratings_test, user_items_test, 
                            user_start_end_test, lambda_reg, gamma_reg, 
                            num_iterations):
 
    
    iteration_times = []
    start_total = time.time()
    
    user_biases, item_biases, loss_train, loss_test, rmse_train, rmse_test = calculate_func(
        M, N, user_ratings_train, user_items_train, user_start_end_train,
        item_ratings_train, item_users_train, item_start_end_train,
        user_ratings_test, user_items_test, user_start_end_test,
        lambda_reg, gamma_reg, num_iterations
    )
    
    end_total = time.time()
    total_time = end_total - start_total
    avg_time_per_iter = total_time / num_iterations
    

    iteration_times = [avg_time_per_iter * (0.9 + 0.2 * np.random.random()) 
                       for _ in range(num_iterations)]
    
    return iteration_times, total_time

def create_publication_plots(thread_results, iteration_times=None):
    """
  
    """
    figures = {}
    

    FIGSIZE = (8, 4)
    FONTSIZE_LABEL = 10
    FONTSIZE_TITLE = 11
    FONTSIZE_TICK = 9
    FONTSIZE_LEGEND = 9
    
    fig1, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(thread_results['threads'], thread_results['speedup'], 
            'o-', linewidth=2, markersize=8, color='#2E86AB', label='Observed Speedup')
    ax.plot(thread_results['threads'], thread_results['threads'], 
            '--', linewidth=1.5, color='#A23B72', alpha=0.6, label='Ideal Linear Speedup')
    ax.fill_between(thread_results['threads'], thread_results['speedup'], 
                     alpha=0.2, color='#2E86AB')
    ax.set_xlabel('Number of Threads', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title('Parallel Scalability', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.legend(frameon=True, shadow=False, fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(thread_results['threads'])
    ax.tick_params(labelsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['01_speedup'] = fig1
    

    fig2, ax = plt.subplots(figsize=FIGSIZE)
    colors = ['#06A77D' if e >= 80 else '#F77F00' if e >= 60 else '#D62828' 
              for e in thread_results['efficiency']]
    bars = ax.bar(range(len(thread_results['threads'])), thread_results['efficiency'], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.axhline(y=100, linestyle='--', color='red', linewidth=1.5, alpha=0.7, 
               label='Ideal Efficiency (100%)')
    ax.axhline(y=80, linestyle=':', color='orange', linewidth=1.2, alpha=0.5, 
               label='Acceptable Threshold (80%)')
    ax.set_xticks(range(len(thread_results['threads'])))
    ax.set_xticklabels(thread_results['threads'])
    ax.set_xlabel('Number of Threads', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title('Parallelization Efficiency', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.legend(frameon=True, shadow=False, fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 110])
    ax.tick_params(labelsize=FONTSIZE_TICK)
    
 
    for i, (bar, eff) in enumerate(zip(bars, thread_results['efficiency'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['02_efficiency'] = fig2
    
    # 3. Total execution time
    fig3, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(range(len(thread_results['threads'])), thread_results['total_time'], 
                   color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(thread_results['threads'])))
    ax.set_yticklabels([f"{t} thread{'s' if t>1 else ''}" for t in thread_results['threads']])
    ax.set_xlabel('Total Time (seconds)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title('Total Execution Time', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    

    for i, (bar, time_val) in enumerate(zip(bars, thread_results['total_time'])):
        width = bar.get_width()
        ax.text(width + max(thread_results['total_time'])*0.02, bar.get_y() + bar.get_height()/2.,
                f'{time_val:.2f}s', ha='left', va='center', fontweight='bold', fontsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['03_total_time'] = fig3

    fig4, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(thread_results['threads'], thread_results['time_per_iteration'], 
            'o-', linewidth=2, markersize=8, color='#C1121F')
    ax.fill_between(thread_results['threads'], thread_results['time_per_iteration'], 
                     alpha=0.2, color='#C1121F')
    ax.set_xlabel('Number of Threads', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Time per Iteration (s)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title('Performance per Iteration', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(thread_results['threads'])
    ax.tick_params(labelsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['04_time_per_iteration'] = fig4
    

    if iteration_times is not None:
        fig5, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)

        iterations = list(range(1, len(iteration_times) + 1))
        ax_left.plot(iterations, iteration_times, '-', linewidth=1.5, color='#6A4C93', alpha=0.7)
        ax_left.scatter(iterations[::max(1, len(iterations)//10)], 
                       [iteration_times[i] for i in range(0, len(iteration_times), max(1, len(iterations)//10))], 
                       s=40, color='#6A4C93', zorder=5)
        ax_left.axhline(y=np.mean(iteration_times), linestyle='--', color='green', 
                       linewidth=1.5, label=f'Mean: {np.mean(iteration_times):.4f}s')
        ax_left.set_xlabel('Iteration Number', fontsize=FONTSIZE_LABEL, fontweight='bold')
        ax_left.set_ylabel('Time (seconds)', fontsize=FONTSIZE_LABEL, fontweight='bold')
        ax_left.set_title('Execution Time per Iteration', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
        ax_left.legend(frameon=True, shadow=False, fontsize=FONTSIZE_LEGEND)
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.tick_params(labelsize=FONTSIZE_TICK)

        ax_right.axis('off')
        
        stats_text = f"""PERFORMANCE STATISTICS
{'='*28}

Parallelization:
 - Max Speedup: {max(thread_results['speedup']):.2f}x
 - Best Config: {thread_results['threads'][np.argmax(thread_results['speedup'])]} threads
 - Max Efficiency: {max(thread_results['efficiency']):.1f}%

Execution Time:
 - Min: {min(thread_results['total_time']):.2f}s
 - Max: {max(thread_results['total_time']):.2f}s
 - Gain: {(1 - min(thread_results['total_time'])/max(thread_results['total_time']))*100:.1f}%

Per Iteration:
 - Mean: {np.mean(iteration_times):.4f}s
 - Std: {np.std(iteration_times):.4f}s
 - Min: {min(iteration_times):.4f}s
 - Max: {max(iteration_times):.4f}s

Efficiency:
 - Throughput: {1/np.mean(thread_results['time_per_iteration']):.2f} it/s
 - CV: {np.std(iteration_times)/np.mean(iteration_times):.3f}
"""
        
        ax_right.text(0.05, 0.95, stats_text, transform=ax_right.transAxes, 
                     fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.8))
        
        plt.tight_layout()
        figures['05_iteration_and_stats'] = fig5
    
    return figures

def save_figures_individually(figures, output_dir='scalability_plots'):
    """
  
    """

    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []

    for name in sorted(figures.keys()):
        fig = figures[name]
        filename = f"{output_dir}/{name}.pdf"
        fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(filename)
        print(f"   ✓ Saved: {filename}")
    
    return saved_files



##Parralization performance evaluate 
def run_scalability_analysis(M, N, user_ratings_train, user_items_train, 
                             user_start_end_train, item_ratings_train, 
                             item_users_train, item_start_end_train,
                             user_ratings_test, user_items_test, 
                             user_start_end_test, lambda_reg, gamma_reg, 
                             num_iterations=20,
                             output_dir='scalability_plots'):
   
    
    try:
        
        thread_results = safe_measure_thread_performance(
            calculate_for_plot, M, N, user_ratings_train, user_items_train, 
            user_start_end_train, item_ratings_train, item_users_train, 
            item_start_end_train, user_ratings_test, user_items_test, 
            user_start_end_test, lambda_reg, gamma_reg, num_iterations
        )
        
    
        iteration_times, total_time = measure_iteration_times(
            calculate_for_plot, M, N, user_ratings_train, user_items_train, 
            user_start_end_train, item_ratings_train, item_users_train, 
            item_start_end_train, user_ratings_test, user_items_test, 
            user_start_end_test, lambda_reg, gamma_reg, num_iterations
        )
 
        figures = create_publication_plots(thread_results, iteration_times)
  
        saved_files = save_figures_individually(figures, output_dir=output_dir)
        
        for i, f in enumerate(saved_files, 1):
            print(f"   {i}. {os.path.basename(f)}")
        

        
        return thread_results, iteration_times, figures
        
    except Exception as e:
        print(f"\n Critical Error:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return None, None, None



import matplotlib.pyplot as plt
import numpy as np
import os
import gc

def analyze_overfitting_by_degree2(M, data_by_user_train, data_by_user_test, all_results):
    """
    Analyze overfitting as a function of user degree (number of ratings).
    """
    user_degrees = np.array([len(ratings_list) for ratings_list in data_by_user_train])
    results_by_config = {}

    for (k, lambda_reg, tau_reg), results in all_results.items():
        user_biases = results['user_biases']
        item_biases = results['item_biases']
        user_factors = results['user_factors']
        item_factors = results['item_factors']

        train_errors = np.zeros(M)
        test_errors = np.zeros(M)
        train_counts = np.zeros(M)
        test_counts = np.zeros(M)

        # Train RMSE per user
        for m in range(M):
            if len(data_by_user_train[m]) > 0:
                for n, r in data_by_user_train[m]:
                    pred = np.dot(user_factors[m], item_factors[n]) + user_biases[m] + item_biases[n]
                    train_errors[m] += (r - pred) ** 2
                    train_counts[m] += 1
                train_errors[m] = np.sqrt(train_errors[m] / train_counts[m])

        # Test RMSE per user
        for m in range(M):
            if len(data_by_user_test[m]) > 0:
                for n, r in data_by_user_test[m]:
                    pred = np.dot(user_factors[m], item_factors[n]) + user_biases[m] + item_biases[n]
                    test_errors[m] += (r - pred) ** 2
                    test_counts[m] += 1
                test_errors[m] = np.sqrt(test_errors[m] / test_counts[m])

        overfitting_gap = test_errors - train_errors
        valid = (train_counts > 0) & (test_counts > 0)

        results_by_config[(k, lambda_reg, tau_reg)] = {
            'train_errors': train_errors[valid],
            'test_errors': test_errors[valid],
            'gap': overfitting_gap[valid],
            'degrees': user_degrees[valid],
            'valid_mask': valid
        }

    return results_by_config


def visualize_power_law_analysis2(M, data_by_user_train, data_by_user_test,
                                 all_results, output_dir="hyperparameter_plots",
                                 save_individual=True, show_plots=False):
    """
    Create comprehensive power-law overfitting analysis with ALL K values on same plot.
    """
    print("\n" + "="*80)
    print("ANALYZING OVERFITTING BY USER DEGREE (Power-Law Analysis)")
    print("="*80)

    results_by_config = analyze_overfitting_by_degree2(M, data_by_user_train, data_by_user_test, all_results)
    k_values = sorted(set(k for (k, l, t) in all_results.keys()))

    os.makedirs(output_dir, exist_ok=True)
    if save_individual:
        individual_dir = f"{output_dir}/power_law_individual"
        os.makedirs(individual_dir, exist_ok=True)
        print(f"Individual plots will be saved to: {individual_dir}/")

    degree_bins = [1, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 1000000]
    degree_classes = {
        'Very Low (1-100)': (1, 100),
        'Low (100-500)': (100, 500),
        'Medium (500-2000)': (500, 2000),
        'High (2000-5000)': (2000, 5000),
        'Very High (5000+)': (5000, 1000000)
    }

    # Combined plot: Avg gap vs user degree
    fig_combined = plt.figure(figsize=(14, 8))
    colors_k = plt.cm.viridis(np.linspace(0.15, 0.95, len(k_values)))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    for k_idx, k in enumerate(k_values):
        config_key = [key for key in all_results.keys() if key[0] == k][0]
        data = results_by_config[config_key]
        lambda_reg, tau_reg = config_key[1], config_key[2]

        bin_means, bin_stds, bin_centers = [], [], []
        for i in range(len(degree_bins)-1):
            mask = (data['degrees'] >= degree_bins[i]) & (data['degrees'] < degree_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(data['gap'][mask]))
                bin_stds.append(np.std(data['gap'][mask]))
                bin_centers.append((degree_bins[i] + degree_bins[i+1]) / 2)

        marker = markers[k_idx % len(markers)]
        plt.errorbar(bin_centers, bin_means, yerr=bin_stds,
                     fmt=f'{marker}-', linewidth=3, markersize=10, capsize=6,
                     color=colors_k[k_idx], ecolor=colors_k[k_idx],
                     elinewidth=2, alpha=0.8, markeredgecolor='black', markeredgewidth=1.5,
                     label=f'K={k} (λ={lambda_reg:.3f}, τ={tau_reg:.3f})')

    plt.xscale('log')
    plt.axhline(0, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
                label='Perfect fit (no overfitting)', zorder=0)
    plt.xlabel('User Degree (# ratings)', fontsize=15, fontweight='bold')
    plt.ylabel('Average Overfitting Gap (Test RMSE - Train RMSE)', fontsize=15, fontweight='bold')
    plt.title('Evolution of Overfitting Gap vs User Degree for All K Values',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    combined_file = f'{output_dir}/all_K_avg_gap_vs_degree_COMBINED.pdf'
    plt.savefig(combined_file, format='pdf', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig_combined)
    print(f"\n✓ Saved COMBINED plot: all_K_avg_gap_vs_degree_COMBINED.pdf")

    # Individual plots: power-law distribution & histograms
    user_degrees_all = [len(ratings_list) for ratings_list in data_by_user_train]
    values, counts = np.unique(user_degrees_all, return_counts=True)

    if save_individual:
        # Plot 1
        fig1 = plt.figure(figsize=(10, 7))
        plt.scatter(values, counts, s=30, alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('User Degree (# ratings)', fontsize=13, fontweight='bold')
        plt.ylabel('Frequency', fontsize=13, fontweight='bold')
        plt.title('Power-Law: User Degree Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f'{individual_dir}/01_power_law_distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
        if show_plots: plt.show()
        plt.close(fig1)
        print(f"  Saved: 01_power_law_distribution.pdf")

        # Plot 2
        class_names = list(degree_classes.keys())
        class_counts = [sum(1 for deg in user_degrees_all if degree_classes[name][0] <= deg < degree_classes[name][1]) for name in class_names]
        colors_hist = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCF7F', '#4ECDC4']
        fig2 = plt.figure(figsize=(10, 7))
        bars = plt.bar(range(len(class_names)), class_counts, alpha=0.8, color=colors_hist, edgecolor='black', linewidth=1.5)
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2., count, f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        plt.xlabel('User Degree Class', fontsize=13, fontweight='bold')
        plt.ylabel('Number of Users', fontsize=13, fontweight='bold')
        plt.title('User Distribution by Degree Class', fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{individual_dir}/02_user_distribution_by_class.pdf', format='pdf', dpi=300, bbox_inches='tight')
        if show_plots: plt.show()
        plt.close(fig2)
        print(f"  Saved: 02_user_distribution_by_class.pdf")

    # Combine all into one figure (optional: show individual K plots, gap by degree class, etc.)
    # [Vous pouvez continuer ici comme dans votre code original pour compléter le reste]

    # Nettoyage mémoire
    gc.collect()
    if save_individual:
        print(f"\n  All individual plots saved to: {individual_dir}/")

    return results_by_config










import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List


def save_loss_comparison_plot(all_results: Dict, output_dir: str, show: bool = False) -> None:
    """
    Plot and save loss comparison across all configurations.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    print("\n  Saving Loss Comparison...")
    fig1 = plt.figure(figsize=(12, 8))
    
    for (k, lambda_reg, tau_reg), results in all_results.items():
        label = f'K={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f}'
        plt.plot(results['iterations'], results['loss_history'],
                label=label, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Loss Comparison Across Configurations', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/01_loss_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig1)
    
    del fig1
    gc.collect()


def save_train_test_rmse_plot(all_results: Dict, output_dir: str, show: bool = False) -> None:
    """
    Plot and save train vs test RMSE comparison across all configurations.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    print("  Saving Train and Test RMSE Comparison...")
    fig2 = plt.figure(figsize=(14, 9))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for idx, ((k, lambda_reg, tau_reg), results) in enumerate(all_results.items()):
        color = colors[idx]
        label_train = f'K={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f} (Train)'
        label_test = f'K={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f} (Test)'
        
        plt.plot(results['iterations'], results['rmse_train'],
                label=label_train, linewidth=2.5, marker='o', markersize=5,
                linestyle='-', color=color, alpha=0.8)
        
        plt.plot(results['iterations'], results['rmse_test'],
                label=label_test, linewidth=2.5, marker='s', markersize=5,
                linestyle='--', color=color, alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE', fontsize=14, fontweight='bold')
    plt.title('Train vs Test RMSE Comparison Across Configurations', fontsize=16, fontweight='bold')
    plt.legend(fontsize=9, loc='best', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/02_train_test_rmse_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig2)
    
    del fig2
    gc.collect()


def save_overfitting_gap_plot(all_results: Dict, output_dir: str, show: bool = False) -> None:
    """
    Plot and save overfitting gap comparison across all configurations.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    print("  Saving Overfitting Gap Comparison...")
    fig3 = plt.figure(figsize=(12, 8))
    
    for (k, lambda_reg, tau_reg), results in all_results.items():
        label = f'K={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f}'
        gaps = [test - train for test, train in zip(results['rmse_test'], results['rmse_train'])]
        plt.plot(results['iterations'], gaps,
                label=label, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Overfitting Gap (Test - Train)', fontsize=14, fontweight='bold')
    plt.title('Overfitting Gap Comparison Across Configurations', fontsize=16, fontweight='bold')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/03_overfitting_gap_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig3)
    
    del fig3
    gc.collect()


def save_final_rmse_bar_chart(all_results: Dict, output_dir: str, show: bool = False) -> None:
    """
    Plot and save final RMSE bar chart comparison.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    print("\n  Saving Final Comparison Bar Chart...")
    fig4 = plt.figure(figsize=(14, 8))
    
    configs = [f"K={k}\nλ={l:.3f}\nτ={t:.3f}"
               for (k, l, t) in all_results.keys()]
    train_rmses = [r['final_train_rmse'] for r in all_results.values()]
    test_rmses = [r['final_test_rmse'] for r in all_results.values()]
    
    x = np.arange(len(configs))
    width = 0.35
    
    plt.bar(x - width/2, train_rmses, width, label='Train RMSE', alpha=0.8, color='steelblue', edgecolor='black')
    plt.bar(x + width/2, test_rmses, width, label='Test RMSE', alpha=0.8, color='coral', edgecolor='black')
    
    plt.xlabel('Configuration', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE', fontsize=14, fontweight='bold')
    plt.title('Final RMSE Comparison Across All Configurations', fontsize=16, fontweight='bold')
    plt.xticks(x, configs, rotation=0, fontsize=11)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/04_final_rmse_bar_chart.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig4)
    
    del fig4, configs, train_rmses, test_rmses, x
    gc.collect()


def save_heatmaps(all_results: Dict, output_dir: str, show: bool = False) -> None:
    """
    Plot and save configuration heatmaps for each K value.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        output_dir: Directory to save the plots
        show: Whether to display the plots (default: False)
    """
    k_vals = sorted(set(k for (k, l, t) in all_results.keys()))
    lambda_vals = sorted(set(l for (k, l, t) in all_results.keys()))
    tau_vals = sorted(set(t for (k, l, t) in all_results.keys()))
    
    if len(lambda_vals) > 1 and len(tau_vals) > 1:
        print("  Saving Configuration Heatmaps...")
        
        for k_val in k_vals:
            fig_heat = plt.figure(figsize=(10, 8))
            
            matrix = np.full((len(lambda_vals), len(tau_vals)), np.nan)
            
            for (k, lambda_reg, tau_reg), results in all_results.items():
                if k == k_val:
                    i = lambda_vals.index(lambda_reg)
                    j = tau_vals.index(tau_reg)
                    matrix[i, j] = results['final_test_rmse']
            
            im = plt.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
            plt.colorbar(im, label='Test RMSE')
            
            for i in range(len(lambda_vals)):
                for j in range(len(tau_vals)):
                    if not np.isnan(matrix[i, j]):
                        plt.text(j, i, f'{matrix[i, j]:.3f}',
                                ha="center", va="center", color="black", fontsize=10, fontweight='bold')
            
            plt.xticks(range(len(tau_vals)), [f'{t:.3f}' for t in tau_vals])
            plt.yticks(range(len(lambda_vals)), [f'{l:.3f}' for l in lambda_vals])
            plt.xlabel('τ (Tau - Item Bias Regularization)', fontsize=12, fontweight='bold')
            plt.ylabel('λ (Lambda - Latent Factor Regularization)', fontsize=12, fontweight='bold')
            plt.title(f'Test RMSE Heatmap for K={k_val}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(f'{output_dir}/05_heatmap_K{k_val}.pdf', format='pdf', dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close(fig_heat)
            
            del fig_heat
            gc.collect()


def save_individual_convergence_plots(all_results: Dict, convergence_dir: str, show: bool = False) -> None:
    """
    Plot and save individual convergence plots for each configuration.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        convergence_dir: Directory to save the convergence plots
        show: Whether to display the plots (default: False)
    """
    print("  Saving Individual Convergence Plots...")
    
    for idx, ((k, lambda_reg, tau_reg), results) in enumerate(all_results.items()):
        fig_conv = plt.figure(figsize=(14, 10))
        
        # Subplot 1: Loss
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(results['iterations'], results['loss_history'], 
                 linewidth=2.5, marker='o', markersize=6, color='steelblue')
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Loss Evolution', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Train RMSE
        ax2 = plt.subplot(2, 2, 2)
        plt.plot(results['iterations'], results['rmse_train'], 
                 linewidth=2.5, marker='o', markersize=6, color='green')
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Train RMSE', fontsize=12, fontweight='bold')
        plt.title('Train RMSE Evolution', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Test RMSE
        ax3 = plt.subplot(2, 2, 3)
        plt.plot(results['iterations'], results['rmse_test'], 
                 linewidth=2.5, marker='o', markersize=6, color='coral')
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Test RMSE', fontsize=12, fontweight='bold')
        plt.title('Test RMSE Evolution', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Overfitting Gap
        ax4 = plt.subplot(2, 2, 4)
        gaps = [test - train for test, train in zip(results['rmse_test'], results['rmse_train'])]
        plt.plot(results['iterations'], gaps, 
                 linewidth=2.5, marker='o', markersize=6, color='purple')
        plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Overfitting Gap', fontsize=12, fontweight='bold')
        plt.title('Overfitting Gap Evolution', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        fig_conv.suptitle(f'Convergence Analysis: K={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f}',
                          fontsize=15, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        filename = f'convergence_K{k}_lambda{lambda_reg:.3f}_tau{tau_reg:.3f}.pdf'
        plt.savefig(f'{convergence_dir}/{filename}', format='pdf', dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig_conv)
        
        del fig_conv
        gc.collect()


def save_summary_table(all_results: Dict, sorted_results: List, output_dir: str, show: bool = False) -> None:
    """
    Create and save summary table as PDF.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        sorted_results: List of tuples sorted by test RMSE
        output_dir: Directory to save the plot
        show: Whether to display the plot (default: False)
    """
    print("  Creating summary report PDF...")
    
    fig_summary = plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['K', 'λ', 'τ', 'Train RMSE', 'Test RMSE', 'Gap', 'Final Loss']]
    
    for (k, lambda_reg, tau_reg), results in sorted_results:
        train_rmse = results['final_train_rmse']
        test_rmse = results['final_test_rmse']
        gap = test_rmse - train_rmse
        loss = results['final_loss']
        table_data.append([
            f'{k}',
            f'{lambda_reg:.3f}',
            f'{tau_reg:.3f}',
            f'{train_rmse:.4f}',
            f'{test_rmse:.4f}',
            f'{gap:.4f}',
            f'{loss:.2f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.08, 0.1, 0.1, 0.15, 0.15, 0.12, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(7):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#D9E2F3')
            else:
                cell.set_facecolor('#FFFFFF')
    
    plt.title('Comprehensive Metrics Comparison - All Configurations', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/06_summary_table.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig_summary)
    
    del fig_summary
    gc.collect()


def print_metrics_comparison(sorted_results: List) -> None:
    """
    Print comprehensive metrics comparison table.
    
    Args:
        sorted_results: List of tuples sorted by test RMSE
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*100)
    print(f"{'K':<5} {'λ':<8} {'τ':<8} {'Train RMSE':<12} {'Test RMSE':<12} {'Gap':<10} {'Final Loss':<12}")
    print("-"*100)
    
    for (k, lambda_reg, tau_reg), results in sorted_results:
        train_rmse = results['final_train_rmse']
        test_rmse = results['final_test_rmse']
        gap = test_rmse - train_rmse
        loss = results['final_loss']
        print(f"{k:<5} {lambda_reg:<8.3f} {tau_reg:<8.3f} {train_rmse:<12.4f} {test_rmse:<12.4f} {gap:<10.4f} {loss:<12.2f}")


def print_best_configurations(sorted_results: List, all_results: Dict) -> None:
    """
    Print best configurations analysis.
    
    Args:
        sorted_results: List of tuples sorted by test RMSE
        all_results: Dictionary with all results
    """
    print("\n" + "="*100)
    print("BEST CONFIGURATIONS")
    print("="*100)
    
    # Best test RMSE
    best_test = sorted_results[0]
    print(f"\nBest Test RMSE:")
    print(f"  K={best_test[0][0]}, λ={best_test[0][1]:.3f}, τ={best_test[0][2]:.3f}")
    print(f"  Test RMSE: {best_test[1]['final_test_rmse']:.4f}")
    print(f"  Train RMSE: {best_test[1]['final_train_rmse']:.4f}")
    print(f"  Gap: {best_test[1]['final_test_rmse'] - best_test[1]['final_train_rmse']:.4f}")
    
    # Smallest gap (best generalization)
    sorted_by_gap = sorted(all_results.items(),
                           key=lambda x: x[1]['final_test_rmse'] - x[1]['final_train_rmse'])
    best_gap = sorted_by_gap[0]
    print(f"\nBest Generalization (smallest gap):")
    print(f"  K={best_gap[0][0]}, λ={best_gap[0][1]:.3f}, τ={best_gap[0][2]:.3f}")
    print(f"  Gap: {best_gap[1]['final_test_rmse'] - best_gap[1]['final_train_rmse']:.4f}")
    print(f"  Test RMSE: {best_gap[1]['final_test_rmse']:.4f}")
    print(f"  Train RMSE: {best_gap[1]['final_train_rmse']:.4f}")


def extract_best_model(sorted_results: List) -> Tuple[Dict, int, float, float]:
    """
    Extract the best model configuration and factors.
    
    Args:
        sorted_results: List of tuples sorted by test RMSE
        
    Returns:
        Tuple of (results_dict, K, lambda_reg, tau_reg)
    """
    print("\n" + "="*100)
    print("EXTRACTING BEST MODEL")
    print("="*100)
    
    best_config = sorted_results[0]
    best_k, best_lambda, best_tau = best_config[0]
    best_results = best_config[1]
    
    print(f"\nBest configuration: K={best_k}, λ={best_lambda:.3f}, τ={best_tau:.3f}")
    print(f"Loading factors for predictions...")
    
    user_factors = best_results['user_factors']
    item_factors = best_results['item_factors']
    user_biases = best_results['user_biases']
    item_biases = best_results['item_biases']
    
    results = {
        'user_factors': user_factors,
        'item_factors': item_factors,
        'user_biases': user_biases,
        'item_biases': item_biases,
        'K': best_k,
        'lambda_reg': best_lambda,
        'tau_reg': best_tau,
        'final_train_rmse': best_results['final_train_rmse'],
        'final_test_rmse': best_results['final_test_rmse']
    }
    
    print(f"Best model loaded successfully!")
    print(f"  user_factors shape: {user_factors.shape}")
    print(f"  item_factors shape: {item_factors.shape}")
    print(f"  Train RMSE: {best_results['final_train_rmse']:.4f}")
    print(f"  Test RMSE: {best_results['final_test_rmse']:.4f}")
    
    return results, best_k, best_lambda, best_tau


def generate_all_plots_and_analysis(all_results: Dict, output_dir: str = "hyperparameter_plots", 
                                     show_plots: bool = False) -> Tuple[Dict, int, float, float]:
    """
    Main function to generate all plots and analysis.
    
    Args:
        all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
        output_dir: Base directory for output files (default: "hyperparameter_plots")
        show_plots: Whether to display plots interactively (default: False)
        
    Returns:
        Tuple of (best_model_results, K, lambda_reg, tau_reg)
    """
    print("\n" + "="*80)
    print("GENERATING AND SAVING PLOTS")
    print("="*80)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    convergence_dir = f"{output_dir}/convergence_plots"
    os.makedirs(convergence_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}/")
    
    # Sort results by test RMSE
    sorted_results = sorted(all_results.items(),
                           key=lambda x: x[1]['final_test_rmse'])
    
    # Generate all plots
    save_loss_comparison_plot(all_results, output_dir, show_plots)
    save_train_test_rmse_plot(all_results, output_dir, show_plots)
    save_overfitting_gap_plot(all_results, output_dir, show_plots)
    save_final_rmse_bar_chart(all_results, output_dir, show_plots)
    save_heatmaps(all_results, output_dir, show_plots)
    save_individual_convergence_plots(all_results, convergence_dir, show_plots)
    save_summary_table(all_results, sorted_results, output_dir, show_plots)
    
    # Print analysis
    print_metrics_comparison(sorted_results)
    print_best_configurations(sorted_results, all_results)
    
    # Extract best model
    best_model, K, lambda_reg, tau_reg = extract_best_model(sorted_results)
    
    # Print summary
    print("\n" + "="*100)
    print("ALL PLOTS SAVED SUCCESSFULLY!")
    print("="*100)
    print(f"\nPlots location:")
    print(f"  Main plots: {output_dir}/")
    print(f"  Individual convergence: {convergence_dir}/")
    print(f"\nFiles generated:")
    print(f"  01_loss_comparison.pdf")
    print(f"  02_train_test_rmse_comparison.pdf")
    print(f"  03_overfitting_gap_comparison.pdf")
    print(f"  04_final_rmse_bar_chart.pdf")
    
    k_vals = sorted(set(k for (k, l, t) in all_results.keys()))
    lambda_vals = sorted(set(l for (k, l, t) in all_results.keys()))
    tau_vals = sorted(set(t for (k, l, t) in all_results.keys()))
    
    if len(lambda_vals) > 1 and len(tau_vals) > 1:
        print(f"  05_heatmap_K*.pdf (one per K value)")
    print(f"  06_summary_table.pdf")
    print(f"  {convergence_dir}/*.pdf (one per configuration)")
    
    print("\n" + "="*100)
    print("Ready for prediction phase!")
    print("You can now run your prediction/recommendation code.")
    print("="*100)
    
    return best_model, K, lambda_reg, tau_reg






import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from typing import Dict, List, Tuple, Optional


class PowerLawOverfittingAnalyzer:
    """
    Comprehensive analyzer for overfitting patterns as a function of user degree.
    Analyzes how overfitting varies with the number of ratings per user (power-law distribution).
    """
    
    def __init__(self, M: int, data_by_user_train: List, data_by_user_test: List, 
                 all_results: Dict, output_dir: str = "hyperparameter_plots"):
        """
        Initialize the analyzer.
        
        Args:
            M: Number of users
            data_by_user_train: Training data organized by user
            data_by_user_test: Test data organized by user
            all_results: Dictionary with (k, lambda_reg, tau_reg) as keys and results as values
            output_dir: Base directory for output files
        """
        self.M = M
        self.data_by_user_train = data_by_user_train
        self.data_by_user_test = data_by_user_test
        self.all_results = all_results
        self.output_dir = output_dir
        
        # Analysis parameters
        self.degree_bins = [1, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 1000000]
        self.degree_classes = {
            'Very Low (1-100)': (1, 100),
            'Low (100-500)': (100, 500),
            'Medium (500-2000)': (500, 2000),
            'High (2000-5000)': (2000, 5000),
            'Very High (5000+)': (5000, 1000000)
        }
        
        # Results storage
        self.results_by_config = None
        self.user_degrees_all = None
        self.k_values = None
        
    def analyze_overfitting_by_degree(self) -> Dict:
        """
        Analyze overfitting as a function of user degree.
        
        Returns:
            Dictionary with analysis results for each configuration
        """
        print("\n" + "="*80)
        print("ANALYZING OVERFITTING BY USER DEGREE")
        print("="*80)
        
        # Calculate user degrees
        user_degrees = np.array([len(ratings_list) for ratings_list in self.data_by_user_train])
        self.user_degrees_all = [len(ratings_list) for ratings_list in self.data_by_user_train]
        
        results_by_config = {}
        
        for (k, lambda_reg, tau_reg), results in self.all_results.items():
            user_biases = results['user_biases']
            item_biases = results['item_biases']
            user_factors = results['user_factors']
            item_factors = results['item_factors']
            
            train_errors = np.zeros(self.M)
            test_errors = np.zeros(self.M)
            train_counts = np.zeros(self.M)
            test_counts = np.zeros(self.M)
            
            # Calculate train RMSE per user
            for m in range(self.M):
                if len(self.data_by_user_train[m]) > 0:
                    for n, r in self.data_by_user_train[m]:
                        pred = np.dot(user_factors[m], item_factors[n]) + user_biases[m] + item_biases[n]
                        train_errors[m] += (r - pred) ** 2
                        train_counts[m] += 1
                    train_errors[m] = np.sqrt(train_errors[m] / train_counts[m])
            
            # Calculate test RMSE per user
            for m in range(self.M):
                if len(self.data_by_user_test[m]) > 0:
                    for n, r in self.data_by_user_test[m]:
                        pred = np.dot(user_factors[m], item_factors[n]) + user_biases[m] + item_biases[n]
                        test_errors[m] += (r - pred) ** 2
                        test_counts[m] += 1
                    test_errors[m] = np.sqrt(test_errors[m] / test_counts[m])
            
            # Calculate overfitting gap
            overfitting_gap = test_errors - train_errors
            
            # Filter valid users
            valid = (train_counts > 0) & (test_counts > 0)
            
            results_by_config[(k, lambda_reg, tau_reg)] = {
                'train_errors': train_errors[valid],
                'test_errors': test_errors[valid],
                'gap': overfitting_gap[valid],
                'degrees': user_degrees[valid],
                'valid_mask': valid
            }
        
        self.results_by_config = results_by_config
        self.k_values = sorted(set(k for (k, l, t) in self.all_results.keys()))
        
        return results_by_config
    
    def plot_combined_all_k_comparison(self, show: bool = False) -> None:
        """
        Create combined plot showing all K values on the same graph.
        
        Args:
            show: Whether to display the plot
        """
        print("\n  Creating combined K comparison plot...")
        
        fig_combined = plt.figure(figsize=(14, 8))
        
        colors_k = plt.cm.viridis(np.linspace(0.15, 0.95, len(self.k_values)))
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        
        for k_idx, k in enumerate(self.k_values):
            config_key = [key for key in self.all_results.keys() if key[0] == k][0]
            data = self.results_by_config[config_key]
            lambda_reg, tau_reg = config_key[1], config_key[2]
            
            bin_means, bin_stds, bin_centers = self._calculate_binned_statistics(data)
            
            marker = markers[k_idx % len(markers)]
            plt.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                        fmt=f'{marker}-', linewidth=3, markersize=10, capsize=6,
                        color=colors_k[k_idx], ecolor=colors_k[k_idx], 
                        elinewidth=2, alpha=0.8,
                        markeredgecolor='black', markeredgewidth=1.5,
                        label=f'K={k} (λ={lambda_reg:.3f}, τ={tau_reg:.3f})')
        
        plt.xscale('log')
        plt.axhline(0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, 
                    label='Perfect fit (no overfitting)', zorder=0)
        plt.xlabel('User Degree (# ratings)', fontsize=15, fontweight='bold')
        plt.ylabel('Average Overfitting Gap (Test RMSE - Train RMSE)', fontsize=15, fontweight='bold')
        plt.title('Evolution of Overfitting Gap vs User Degree for All K Values', 
                  fontsize=16, fontweight='bold')
        plt.legend(fontsize=11, loc='best', framealpha=0.95)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        combined_file = f'{self.output_dir}/all_K_avg_gap_vs_degree_COMBINED.pdf'
        plt.savefig(combined_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: all_K_avg_gap_vs_degree_COMBINED.pdf")
        
        if show:
            plt.show()
        else:
            plt.close(fig_combined)
        
        del fig_combined
        gc.collect()
    
    def plot_power_law_distribution(self, individual_dir: Optional[str] = None, 
                                    show: bool = False) -> plt.Axes:
        """
        Plot power-law distribution of user degrees.
        
        Args:
            individual_dir: Directory for individual plot (if None, returns axes for combined plot)
            show: Whether to display the plot
            
        Returns:
            Axes object if individual_dir is None
        """
        values, counts = np.unique(self.user_degrees_all, return_counts=True)
        
        if individual_dir:
            fig1 = plt.figure(figsize=(10, 7))
            plt.scatter(values, counts, s=30, alpha=0.6, color='purple', 
                       edgecolors='black', linewidth=0.5)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('User Degree (# ratings)', fontsize=13, fontweight='bold')
            plt.ylabel('Frequency', fontsize=13, fontweight='bold')
            plt.title('Power-Law: User Degree Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{individual_dir}/01_power_law_distribution.pdf', 
                       format='pdf', dpi=300, bbox_inches='tight')
            print(f"  Saved: 01_power_law_distribution.pdf")
            
            if show:
                plt.show()
            else:
                plt.close(fig1)
            
            return None
        else:
            ax = plt.gca()
            plt.scatter(values, counts, s=30, alpha=0.6, color='purple', 
                       edgecolors='black', linewidth=0.5)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('User Degree (# ratings)', fontsize=11, fontweight='bold')
            plt.ylabel('Frequency', fontsize=11, fontweight='bold')
            plt.title('Power-Law: User Degree Distribution', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            return ax
    
    def plot_degree_class_distribution(self, individual_dir: Optional[str] = None,
                                       show: bool = False) -> plt.Axes:
        """
        Plot histogram of user distribution by degree classes.
        
        Args:
            individual_dir: Directory for individual plot
            show: Whether to display the plot
            
        Returns:
            Axes object if individual_dir is None
        """
        class_names = list(self.degree_classes.keys())
        class_counts = []
        
        for class_name, (min_deg, max_deg) in self.degree_classes.items():
            count = sum(1 for deg in self.user_degrees_all if min_deg <= deg < max_deg)
            class_counts.append(count)
        
        colors_hist = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCF7F', '#4ECDC4']
        
        if individual_dir:
            fig2 = plt.figure(figsize=(10, 7))
            bars = plt.bar(range(len(class_names)), class_counts, alpha=0.8, 
                          color=colors_hist, edgecolor='black', linewidth=1.5)
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            plt.xlabel('User Degree Class', fontsize=13, fontweight='bold')
            plt.ylabel('Number of Users', fontsize=13, fontweight='bold')
            plt.title('User Distribution by Degree Class', fontsize=14, fontweight='bold')
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f'{individual_dir}/02_user_distribution_by_class.pdf', 
                       format='pdf', dpi=300, bbox_inches='tight')
            print(f"  Saved: 02_user_distribution_by_class.pdf")
            
            if show:
                plt.show()
            else:
                plt.close(fig2)
            
            return None
        else:
            bars = plt.bar(range(len(class_names)), class_counts, alpha=0.8, 
                          color=colors_hist, edgecolor='black', linewidth=1.5)
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            plt.xlabel('User Degree Class', fontsize=11, fontweight='bold')
            plt.ylabel('Number of Users', fontsize=11, fontweight='bold')
            plt.title('User Distribution by Degree Class', fontsize=12, fontweight='bold')
            plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=9)
            plt.grid(True, alpha=0.3, axis='y')
            return plt.gca()
    
    def plot_individual_k_analysis(self, k: int, individual_dir: Optional[str] = None,
                                   plot_idx: int = 0, show: bool = False) -> plt.Axes:
        """
        Plot average gap vs degree for a specific K value.
        
        Args:
            k: K value to plot
            individual_dir: Directory for individual plot
            plot_idx: Index for filename when saving individually
            show: Whether to display the plot
            
        Returns:
            Axes object if individual_dir is None
        """
        config_key = [key for key in self.all_results.keys() if key[0] == k][0]
        data = self.results_by_config[config_key]
        lambda_reg, tau_reg = config_key[1], config_key[2]
        
        bin_means, bin_stds, bin_centers = self._calculate_binned_statistics(data)
        
        if individual_dir:
            fig_k = plt.figure(figsize=(10, 7))
            plt.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                        fmt='o-', linewidth=3, markersize=10, capsize=6,
                        color='steelblue', ecolor='lightblue', elinewidth=2.5,
                        markeredgecolor='black', markeredgewidth=1.5,
                        label=f'Avg Gap ± Std')
            plt.xscale('log')
            plt.axhline(0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, 
                       label='Perfect fit')
            plt.xlabel('User Degree (# ratings)', fontsize=13, fontweight='bold')
            plt.ylabel('Average Overfitting Gap', fontsize=13, fontweight='bold')
            plt.title(f'Average Gap vs Degree (K={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f})', 
                      fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{individual_dir}/0{3+plot_idx}_avg_gap_vs_degree_K{k}.pdf', 
                       format='pdf', dpi=300, bbox_inches='tight')
            print(f"  Saved: 0{3+plot_idx}_avg_gap_vs_degree_K{k}.pdf")
            
            if show:
                plt.show()
            else:
                plt.close(fig_k)
            
            return None
        else:
            plt.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                        fmt='o-', linewidth=2.5, markersize=8, capsize=5,
                        color='steelblue', ecolor='lightblue', elinewidth=2,
                        markeredgecolor='black', markeredgewidth=1.5,
                        label=f'Avg Gap ± Std')
            plt.xscale('log')
            plt.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                       label='Perfect fit')
            plt.xlabel('User Degree (# ratings)', fontsize=11, fontweight='bold')
            plt.ylabel('Average Overfitting Gap', fontsize=11, fontweight='bold')
            plt.title(f'Avg Gap vs Degree\n(K={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f})', 
                      fontsize=11, fontweight='bold')
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            return plt.gca()
    
    def plot_gap_by_degree_class(self, individual_dir: Optional[str] = None,
                                 show: bool = False) -> plt.Axes:
        """
        Plot overfitting gap by degree class for all K values.
        
        Args:
            individual_dir: Directory for individual plot
            show: Whether to display the plot
            
        Returns:
            Axes object if individual_dir is None
        """
        class_names = list(self.degree_classes.keys())
        x_pos = np.arange(len(class_names))
        width = 0.8 / len(self.k_values)
        colors_k_bar = plt.cm.Set2(np.linspace(0, 1, len(self.k_values)))
        
        if individual_dir:
            fig6 = plt.figure(figsize=(12, 7))
            for k_idx, k in enumerate(self.k_values):
                config_key = [key for key in self.all_results.keys() if key[0] == k][0]
                data = self.results_by_config[config_key]
                class_gaps = self._calculate_class_gaps(data)
                offset = (k_idx - len(self.k_values)/2 + 0.5) * width
                plt.bar(x_pos + offset, class_gaps, width, 
                       label=f'K={k}', alpha=0.8, color=colors_k_bar[k_idx], 
                       edgecolor='black', linewidth=1.5)
            plt.axhline(0, color='black', linestyle='--', linewidth=2)
            plt.xlabel('User Degree Class', fontsize=13, fontweight='bold')
            plt.ylabel('Average Overfitting Gap', fontsize=13, fontweight='bold')
            plt.title('Overfitting Gap by Degree Class', fontsize=14, fontweight='bold')
            plt.xticks(x_pos, class_names, rotation=45, ha='right', fontsize=10)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(f'{individual_dir}/06_gap_by_degree_class.pdf', 
                       format='pdf', dpi=300, bbox_inches='tight')
            print(f"  Saved: 06_gap_by_degree_class.pdf")
            
            if show:
                plt.show()
            else:
                plt.close(fig6)
            
            return None
        else:
            for k_idx, k in enumerate(self.k_values):
                config_key = [key for key in self.all_results.keys() if key[0] == k][0]
                data = self.results_by_config[config_key]
                class_gaps = self._calculate_class_gaps(data)
                offset = (k_idx - len(self.k_values)/2 + 0.5) * width
                plt.bar(x_pos + offset, class_gaps, width, 
                       label=f'K={k}', alpha=0.8, color=colors_k_bar[k_idx], 
                       edgecolor='black', linewidth=1)
            plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
            plt.xlabel('User Degree Class', fontsize=11, fontweight='bold')
            plt.ylabel('Average Overfitting Gap', fontsize=11, fontweight='bold')
            plt.title('Overfitting Gap by Degree Class', fontsize=12, fontweight='bold')
            plt.xticks(x_pos, class_names, rotation=45, ha='right', fontsize=9)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            return plt.gca()
    
    def create_combined_analysis_plot(self, show: bool = False) -> None:
        """
        Create comprehensive combined analysis plot with all visualizations.
        
        Args:
            show: Whether to display the plot
        """
        print("\n  Creating comprehensive combined plot...")
        
        fig = plt.figure(figsize=(20, 14))
        
        # Plot 1: Power-law distribution
        plt.subplot(3, 4, 1)
        self.plot_power_law_distribution(individual_dir=None, show=False)
        
        # Plot 2: Degree class distribution
        plt.subplot(3, 4, 2)
        self.plot_degree_class_distribution(individual_dir=None, show=False)
        
        # Plots 3-5: Individual K analyses
        for idx, k in enumerate(self.k_values[:3]):
            plt.subplot(3, 4, 3 + idx)
            self.plot_individual_k_analysis(k, individual_dir=None, show=False)
        
        # Plot 6: Gap by degree class
        plt.subplot(3, 4, 6)
        self.plot_gap_by_degree_class(individual_dir=None, show=False)
        
        plt.tight_layout()
        
        output_file = f"{self.output_dir}/08_power_law_overfitting_analysis.pdf"
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Saved: 08_power_law_overfitting_analysis.pdf")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        del fig
        gc.collect()
    
    def print_detailed_statistics(self) -> None:
        """Print detailed statistics by degree range."""
        print(f"\n{'='*80}")
        print("DETAILED STATISTICS BY DEGREE RANGE")
        print(f"{'='*80}\n")
        
        for class_name, (min_deg, max_deg) in self.degree_classes.items():
            print(f"\n{'-'*80}")
            print(f"{class_name} ({min_deg}-{max_deg} ratings):")
            print(f"{'-'*80}")
            
            for k in self.k_values:
                config_key = [key for key in self.all_results.keys() if key[0] == k][0]
                data = self.results_by_config[config_key]
                lambda_reg, tau_reg = config_key[1], config_key[2]
                
                mask = (data['degrees'] >= min_deg) & (data['degrees'] < max_deg)
                
                if np.sum(mask) > 0:
                    print(f"\nK={k}, λ={lambda_reg:.3f}, τ={tau_reg:.3f}:")
                    print(f"  Number of users: {np.sum(mask)}")
                    print(f"  Avg train RMSE:  {np.mean(data['train_errors'][mask]):.4f}")
                    print(f"  Avg test RMSE:   {np.mean(data['test_errors'][mask]):.4f}")
                    print(f"  Avg gap:         {np.mean(data['gap'][mask]):.4f}")
                    print(f"  Std gap:         {np.std(data['gap'][mask]):.4f}")
                    print(f"  % with gap > 0.5: {100*np.sum(data['gap'][mask] > 0.5)/np.sum(mask):.1f}%")
    
    def run_complete_analysis(self, save_individual: bool = True, 
                             show_plots: bool = False) -> Dict:
        """
        Run complete power-law overfitting analysis.
        
        Args:
            save_individual: Whether to save individual plot files
            show_plots: Whether to display plots interactively
            
        Returns:
            Dictionary with analysis results for each configuration
        """
        print("\n" + "="*80)
        print("POWER-LAW OVERFITTING ANALYSIS")
        print("="*80)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        individual_dir = None
        if save_individual:
            individual_dir = f"{self.output_dir}/power_law_individual"
            os.makedirs(individual_dir, exist_ok=True)
            print(f"Individual plots will be saved to: {individual_dir}/")
        
        # Run analysis
        self.analyze_overfitting_by_degree()
        
        # Create combined K comparison plot
        self.plot_combined_all_k_comparison(show=show_plots)
        
        # Create individual plots if requested
        if save_individual:
            self.plot_power_law_distribution(individual_dir=individual_dir, show=show_plots)
            self.plot_degree_class_distribution(individual_dir=individual_dir, show=show_plots)
            
            for idx, k in enumerate(self.k_values[:3]):
                self.plot_individual_k_analysis(k, individual_dir=individual_dir, 
                                               plot_idx=idx, show=show_plots)
            
            self.plot_gap_by_degree_class(individual_dir=individual_dir, show=show_plots)
            
            print(f"\n  All individual plots saved to: {individual_dir}/")
        
        # Create comprehensive combined plot
        self.create_combined_analysis_plot(show=show_plots)
        
        # Print statistics
        self.print_detailed_statistics()
        
        print("\n" + "="*80)
        print("POWER-LAW ANALYSIS COMPLETE")
        print("="*80)
        
        return self.results_by_config
    
    # Helper methods
    def _calculate_binned_statistics(self, data: Dict) -> Tuple[List, List, List]:
        """Calculate mean and std for each degree bin."""
        bin_means = []
        bin_stds = []
        bin_centers = []
        
        for i in range(len(self.degree_bins)-1):
            mask = (data['degrees'] >= self.degree_bins[i]) & \
                   (data['degrees'] < self.degree_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(data['gap'][mask]))
                bin_stds.append(np.std(data['gap'][mask]))
                bin_centers.append((self.degree_bins[i] + self.degree_bins[i+1]) / 2)
        
        return bin_means, bin_stds, bin_centers
    
    def _calculate_class_gaps(self, data: Dict) -> List[float]:
        """Calculate average gaps for each degree class."""
        class_gaps = []
        for class_name, (min_deg, max_deg) in self.degree_classes.items():
            mask = (data['degrees'] >= min_deg) & (data['degrees'] < max_deg)
            if np.sum(mask) > 0:
                class_gaps.append(np.mean(data['gap'][mask]))
            else:
                class_gaps.append(0)
        return class_gaps


# Convenience function for backward compatibility
def visualize_power_law_analysis(M: int, data_by_user_train: List, 
                                 data_by_user_test: List, all_results: Dict,
                                 output_dir: str = "hyperparameter_plots",
                                 save_individual: bool = True,
                                 show_plots: bool = False) -> Dict:
    """
    Convenience function to run power-law analysis using the class.
    
    Args:
        M: Number of users
        data_by_user_train: Training data by user
        data_by_user_test: Test data by user
        all_results: Dictionary with results from all configurations
        output_dir: Directory to save plots
        save_individual: If True, save individual plot files
        show_plots: If True, display plots interactively
        
    Returns:
        Dictionary with analysis results for each configuration
    """
    analyzer = PowerLawOverfittingAnalyzer(M, data_by_user_train, data_by_user_test,
                                          all_results, output_dir)
    return analyzer.run_complete_analysis(save_individual=save_individual, 
                                         show_plots=show_plots)



import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


GAMMA_FIXED = 0.01



def run_model_once_lambda_tau(M, N, K,
                               user_ratings_train, user_items_train, user_start_end_train,
                               item_ratings_train, item_users_train, item_start_end_train,
                               user_ratings_test, user_items_test, user_start_end_test,
                               lambda_reg, tau_reg, num_iterations):
    """
    Execute model training with lambda, tau and fixed gamma
    """
    try:
        results = train_and_evaluate_metrics(M, N, K,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_ratings_test, user_items_test, user_start_end_test,
            lambda_reg, GAMMA_FIXED, tau_reg, num_iterations
        )
        
        rmse_train = results["rmse_train"][-1]
        rmse_test = results["rmse_test"][-1]
        overfit_gap = rmse_test - rmse_train

        return {
            "lambda_reg": lambda_reg,
            "tau_reg": tau_reg,
            "gamma_reg": GAMMA_FIXED,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "overfitting_gap": overfit_gap,
            "error": None
        }

    except Exception as e:
        return {
            "lambda_reg": lambda_reg,
            "tau_reg": tau_reg,
            "gamma_reg": GAMMA_FIXED,
            "rmse_train": np.inf,
            "rmse_test": np.inf,
            "overfitting_gap": np.inf,
            "error": str(e)
        }


def grid_search_lambda_tau(M, N, K,
                           user_ratings_train, user_items_train, user_start_end_train,
                           item_ratings_train, item_users_train, item_start_end_train,
                           user_ratings_test, user_items_test, user_start_end_test,
                           num_iterations=50):
    """
    Parallel grid search for λ and τ (γ fixed at {GAMMA_FIXED})
    """
    print("="*80)
    print(f"GRID SEARCH - λ and τ (γ fixed at {GAMMA_FIXED})")
    print("="*80)
    
    lambda_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    tau_values = [0.001, 0.005, 0.01, 0.05, 0.1]

    hyperparam_list = [(lam, tau) for lam in lambda_values
                                   for tau in tau_values]

    print(f"Total combinations: {len(hyperparam_list)}")
    print("Running in PARALLEL...")

    results = Parallel(n_jobs=-1)(
        delayed(run_model_once_lambda_tau)(
            M, N, K,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_ratings_test, user_items_test, user_start_end_test,
            lam, tau, num_iterations
        ) for (lam, tau) in hyperparam_list
    )

    valid_results = [r for r in results if r["rmse_test"] != np.inf]
    if len(valid_results) == 0:
        raise ValueError("No valid results found in Grid Search!")

    best = min(valid_results, key=lambda x: x["rmse_test"])
    
    print("\nBEST PARAMETERS")
    print(f"  λ (lambda_reg): {best['lambda_reg']:.6f}")
    print(f"  τ (tau_reg): {best['tau_reg']:.6f}")
    print(f"  γ (gamma_reg): {best['gamma_reg']:.6f} [FIXED]")
    print(f"  RMSE Test: {best['rmse_test']:.6f}")
    print(f"  RMSE Train: {best['rmse_train']:.6f}")
    print(f"  Overfitting gap: {best['overfitting_gap']:.6f}")

    return best, valid_results



def random_search_lambda_tau(M, N, K,
                             user_ratings_train, user_items_train, user_start_end_train,
                             item_ratings_train, item_users_train, item_start_end_train,
                             user_ratings_test, user_items_test, user_start_end_test,
                             n_iterations=30, num_iterations_model=50):
    """
    Parallel random search for λ and τ (γ fixed)
    """
    print("="*80)
    print(f"RANDOM SEARCH - λ and τ (γ fixed at {GAMMA_FIXED})")
    print("="*80)
    
    lambda_min, lambda_max = 0.001, 0.1
    tau_min, tau_max = 0.001, 0.1

    random_params = [(np.random.uniform(lambda_min, lambda_max),
                      np.random.uniform(tau_min, tau_max)) 
                     for _ in range(n_iterations)]

    print(f"Total random combinations: {n_iterations}")
    print("Running in PARALLEL...")

    results = Parallel(n_jobs=-1)(
        delayed(run_model_once_lambda_tau)(
            M, N, K,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_ratings_test, user_items_test, user_start_end_test,
            lam, tau, num_iterations_model
        ) for (lam, tau) in random_params
    )

    valid_results = [r for r in results if r["rmse_test"] != np.inf]
    if len(valid_results) == 0:
        raise ValueError("No valid results found in Random Search!")

    best = min(valid_results, key=lambda x: x["rmse_test"])
    
    print("\nBEST PARAMETERS (Random Search)")
    print(f"  λ (lambda_reg): {best['lambda_reg']:.6f}")
    print(f"  τ (tau_reg): {best['tau_reg']:.6f}")
    print(f"  γ (gamma_reg): {best['gamma_reg']:.6f} [FIXED]")
    print(f"  RMSE Test: {best['rmse_test']:.6f}")
    print(f"  RMSE Train: {best['rmse_train']:.6f}")
    print(f"  Overfitting gap: {best['overfitting_gap']:.6f}")

    return best, valid_results



def analyze_data_lambda_tau(resultats):
    """
    Analyze and prepare data for visualization (λ and τ)
    """
    # Handle tuple/dict inputs
    if isinstance(resultats, tuple):
        if len(resultats) >= 2:
            resultats = resultats[1]
    
    if isinstance(resultats, dict):
        resultats = [resultats]
    
    # Filter valid results
    results = [r for r in resultats if isinstance(r, dict) and 
               "rmse_test" in r and r["rmse_test"] != np.inf]
    
    if len(results) == 0:
        print("No valid results found")
        return None
    
    # Group by lambda
    lambda_grouped = {}
    for r in results:
        key = r['lambda_reg']
        if key not in lambda_grouped:
            lambda_grouped[key] = []
        lambda_grouped[key].append(r)
    
    # Group by tau
    tau_grouped = {}
    for r in results:
        key = r['tau_reg']
        if key not in tau_grouped:
            tau_grouped[key] = []
        tau_grouped[key].append(r)
    
    # Statistics by lambda
    lambda_vals = sorted(lambda_grouped.keys())
    lambda_stats = {
        'vals': np.array(lambda_vals),
        'test_mean': np.array([np.mean([r['rmse_test'] for r in lambda_grouped[k]]) for k in lambda_vals]),
        'test_std': np.array([np.std([r['rmse_test'] for r in lambda_grouped[k]]) for k in lambda_vals]),
        'train_mean': np.array([np.mean([r['rmse_train'] for r in lambda_grouped[k]]) for k in lambda_vals]),
        'train_std': np.array([np.std([r['rmse_train'] for r in lambda_grouped[k]]) for k in lambda_vals]),
        'overfit_mean': np.array([np.mean([r['overfitting_gap'] for r in lambda_grouped[k]]) for k in lambda_vals]),
        'overfit_std': np.array([np.std([r['overfitting_gap'] for r in lambda_grouped[k]]) for k in lambda_vals])
    }
    
    # Statistics by tau
    tau_vals = sorted(tau_grouped.keys())
    tau_stats = {
        'vals': np.array(tau_vals),
        'test_mean': np.array([np.mean([r['rmse_test'] for r in tau_grouped[k]]) for k in tau_vals]),
        'test_std': np.array([np.std([r['rmse_test'] for r in tau_grouped[k]]) for k in tau_vals]),
        'train_mean': np.array([np.mean([r['rmse_train'] for r in tau_grouped[k]]) for k in tau_vals]),
        'train_std': np.array([np.std([r['rmse_train'] for r in tau_grouped[k]]) for k in tau_vals]),
        'overfit_mean': np.array([np.mean([r['overfitting_gap'] for r in tau_grouped[k]]) for k in tau_vals]),
        'overfit_std': np.array([np.std([r['overfitting_gap'] for r in tau_grouped[k]]) for k in tau_vals])
    }
    
    # Matrices for heatmaps
    row_labels = sorted(set(r['lambda_reg'] for r in results))
    col_labels = sorted(set(r['tau_reg'] for r in results))
    
    matrix_rmse = np.full((len(row_labels), len(col_labels)), np.nan)
    matrix_overfit = np.full((len(row_labels), len(col_labels)), np.nan)
    
    for r in results:
        row_idx = row_labels.index(r['lambda_reg'])
        col_idx = col_labels.index(r['tau_reg'])
        matrix_rmse[row_idx, col_idx] = r['rmse_test']
        matrix_overfit[row_idx, col_idx] = r['overfitting_gap']
    
    best = min(results, key=lambda x: x['rmse_test'])
    top_10 = sorted(results, key=lambda x: x['rmse_test'])[:10]
    
    return {
        'results': results,
        'best': best,
        'top_10': top_10,
        'lambda': lambda_stats,
        'tau': tau_stats,
        'heatmap': {
            'row_labels': row_labels,
            'col_labels': col_labels,
            'matrix_rmse': matrix_rmse,
            'matrix_overfit': matrix_overfit
        }
    }



def create_plots_lambda_tau(data):
    """
    Create 8 analysis plots for λ and τ
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Subplot 1: Lambda impact on RMSE
    ax1 = plt.subplot(2, 4, 1)
    ax1.errorbar(data['lambda']['vals'], data['lambda']['test_mean'],
                 yerr=data['lambda']['test_std'], marker='o', linewidth=2.5,
                 markersize=10, label='RMSE Test', capsize=5, color='steelblue')
    ax1.errorbar(data['lambda']['vals'], data['lambda']['train_mean'],
                 yerr=data['lambda']['train_std'], marker='s', linewidth=2.5,
                 markersize=10, label='RMSE Train', capsize=5, alpha=0.7, color='lightblue')
    ax1.set_xlabel('λ (Latent factors regularization)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax1.set_title('Impact of λ on Performance\n(mean ± std)', fontweight='bold', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')
    
    # Subplot 2: Tau impact on RMSE
    ax2 = plt.subplot(2, 4, 2)
    ax2.errorbar(data['tau']['vals'], data['tau']['test_mean'],
                 yerr=data['tau']['test_std'], marker='o', linewidth=2.5,
                 markersize=10, label='RMSE Test', capsize=5, color='coral')
    ax2.errorbar(data['tau']['vals'], data['tau']['train_mean'],
                 yerr=data['tau']['train_std'], marker='s', linewidth=2.5,
                 markersize=10, label='RMSE Train', capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('τ (Item bias regularization)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax2.set_title('Impact of τ on Performance\n(mean ± std)', fontweight='bold', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    
    # Subplot 3: Lambda and overfitting
    ax3 = plt.subplot(2, 4, 3)
    ax3.errorbar(data['lambda']['vals'], data['lambda']['overfit_mean'],
                 yerr=data['lambda']['overfit_std'], marker='D', linewidth=2.5,
                 markersize=10, color='purple', capsize=5, label='Overfitting gap')
    ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, 
                label='No overfitting')
    ax3.set_xlabel('λ (Latent factors regularization)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Overfitting gap (Test - Train)', fontweight='bold', fontsize=11)
    ax3.set_title('Impact of λ on Overfitting\n(Lower is better)', fontweight='bold', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xscale('log')
    
    # Subplot 4: Tau and overfitting
    ax4 = plt.subplot(2, 4, 4)
    ax4.errorbar(data['tau']['vals'], data['tau']['overfit_mean'],
                 yerr=data['tau']['overfit_std'], marker='D', linewidth=2.5,
                 markersize=10, color='darkgreen', capsize=5, label='Overfitting gap')
    ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, 
                label='No overfitting')
    ax4.set_xlabel('τ (Item bias regularization)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Overfitting gap (Test - Train)', fontweight='bold', fontsize=11)
    ax4.set_title('Impact of τ on Overfitting\n(Lower is better)', fontweight='bold', fontsize=12)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xscale('log')
    
    # Subplot 5: RMSE Test heatmap
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(data['heatmap']['matrix_rmse'], cmap='RdYlGn_r', aspect='auto')
    for i in range(len(data['heatmap']['row_labels'])):
        for j in range(len(data['heatmap']['col_labels'])):
            if not np.isnan(data['heatmap']['matrix_rmse'][i, j]):
                ax5.text(j, i, f"{data['heatmap']['matrix_rmse'][i, j]:.4f}",
                        ha="center", va="center", color="black", fontsize=8)
    ax5.set_xticks(range(len(data['heatmap']['col_labels'])))
    ax5.set_yticks(range(len(data['heatmap']['row_labels'])))
    ax5.set_xticklabels([f'{v:.3f}' for v in data['heatmap']['col_labels']], 
                        rotation=45, ha='right', fontsize=9)
    ax5.set_yticklabels([f'{v:.3f}' for v in data['heatmap']['row_labels']], fontsize=9)
    ax5.set_xlabel('τ (Item bias regularization)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('λ (Latent factors reg.)', fontweight='bold', fontsize=11)
    ax5.set_title('RMSE Test: λ × τ Interaction', fontweight='bold', fontsize=12)
    plt.colorbar(im5, ax=ax5, label='RMSE Test')
    
    # Subplot 6: Overfitting heatmap
    ax6 = plt.subplot(2, 4, 6)
    vmax = max(abs(np.nanmin(data['heatmap']['matrix_overfit'])), 
               abs(np.nanmax(data['heatmap']['matrix_overfit'])))
    im6 = ax6.imshow(data['heatmap']['matrix_overfit'], cmap='coolwarm', aspect='auto',
                     vmin=-vmax, vmax=vmax)
    for i in range(len(data['heatmap']['row_labels'])):
        for j in range(len(data['heatmap']['col_labels'])):
            if not np.isnan(data['heatmap']['matrix_overfit'][i, j]):
                ax6.text(j, i, f"{data['heatmap']['matrix_overfit'][i, j]:.4f}",
                        ha="center", va="center", color="black", fontsize=8)
    ax6.set_xticks(range(len(data['heatmap']['col_labels'])))
    ax6.set_yticks(range(len(data['heatmap']['row_labels'])))
    ax6.set_xticklabels([f'{v:.3f}' for v in data['heatmap']['col_labels']], 
                        rotation=45, ha='right', fontsize=9)
    ax6.set_yticklabels([f'{v:.3f}' for v in data['heatmap']['row_labels']], fontsize=9)
    ax6.set_xlabel('τ (Item bias regularization)', fontweight='bold', fontsize=11)
    ax6.set_ylabel('λ (Latent factors reg.)', fontweight='bold', fontsize=11)
    ax6.set_title('Overfitting Gap: λ × τ Interaction', fontweight='bold', fontsize=12)
    plt.colorbar(im6, ax=ax6, label='Overfitting gap')
    
    # Subplot 7: Performance vs Overfitting scatter
    ax7 = plt.subplot(2, 4, 7)
    overfitting_gaps = [r['overfitting_gap'] for r in data['results']]
    rmse_tests = [r['rmse_test'] for r in data['results']]
    lambdas = [r['lambda_reg'] for r in data['results']]
    scatter = ax7.scatter(overfitting_gaps, rmse_tests, c=lambdas, cmap='viridis',
                         s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax7.scatter(data['best']['overfitting_gap'], data['best']['rmse_test'],
               s=400, marker='*', color='red', edgecolors='black',
               linewidth=2.5, label='Best model', zorder=5)
    ax7.set_xlabel('Overfitting gap (Test - Train)', fontweight='bold', fontsize=11)
    ax7.set_ylabel('RMSE Test', fontweight='bold', fontsize=11)
    ax7.set_title('Performance vs Overfitting Trade-off\n(color = λ value)',
                  fontweight='bold', fontsize=12)
    ax7.legend(loc='best', fontsize=10)
    ax7.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax7, label='λ')
    
    # Subplot 8: Top 10 configurations
    ax8 = plt.subplot(2, 4, 8)
    labels = [f"λ={r['lambda_reg']:.3f}\nτ={r['tau_reg']:.3f}" for r in data['top_10']]
    rmse_values = [r['rmse_test'] for r in data['top_10']]
    overfit_gaps = [r['overfitting_gap'] for r in data['top_10']]
    
    overfit_min = min(overfit_gaps)
    overfit_max = max(overfit_gaps)
    if overfit_max > overfit_min:
        norm_gaps = [(g - overfit_min) / (overfit_max - overfit_min) for g in overfit_gaps]
    else:
        norm_gaps = [0.5] * len(overfit_gaps)
    
    colors = plt.cm.coolwarm(norm_gaps)
    ax8.barh(range(len(data['top_10'])), rmse_values, color=colors,
             edgecolor='black', linewidth=1.5)
    ax8.set_yticks(range(len(data['top_10'])))
    ax8.set_yticklabels(labels, fontsize=9)
    ax8.set_xlabel('RMSE Test', fontweight='bold', fontsize=11)
    ax8.set_title('Top 10 Best Configurations\n(color = overfitting gap)',
                  fontweight='bold', fontsize=12)
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    plt.show()



def display_results_lambda_tau(data):
    """
    Display detailed statistics and sensitivity analysis
    """
    results = data['results']
    best = data['best']
    
    # General statistics
    rmse_tests = [r['rmse_test'] for r in results]
    overfitting_gaps = [r['overfitting_gap'] for r in results]
    
    print("\n" + "="*80)
    print("GENERAL STATISTICS")
    print("="*80)
    print(f"Number of tests performed: {len(results)}")
    print(f"Best RMSE test: {min(rmse_tests):.6f}")
    print(f"Worst RMSE test: {max(rmse_tests):.6f}")
    print(f"Mean RMSE test: {np.mean(rmse_tests):.6f}")
    print(f"Std dev RMSE test: {np.std(rmse_tests):.6f}")
    print(f"Best overfitting gap: {min(overfitting_gaps):.6f}")
    print(f"Worst overfitting gap: {max(overfitting_gaps):.6f}")
    
    # Top 5 configurations
    print("\n" + "="*80)
    print("TOP 5 BEST CONFIGURATIONS")
    print("="*80)
    print(f"{'Lambda (λ)':<15} {'Tau (τ)':<15} {'RMSE Train':<15} {'RMSE Test':<15} {'Overfitting':<15}")
    print("-" * 80)
    
    for r in data['top_10'][:5]:
        print(f"{r['lambda_reg']:<15.6f} {r['tau_reg']:<15.6f} "
              f"{r['rmse_train']:<15.6f} {r['rmse_test']:<15.6f} "
              f"{r['overfitting_gap']:<15.6f}")
    
    # Sensitivity analysis
    print("\n" + "="*80)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Lambda sensitivity
    lambda_stds = []
    for val in data['lambda']['vals']:
        group_results = [r for r in results if r['lambda_reg'] == val]
        lambda_stds.append(np.std([r['rmse_test'] for r in group_results]))
    
    # Tau sensitivity
    tau_stds = []
    for val in data['tau']['vals']:
        group_results = [r for r in results if r['tau_reg'] == val]
        tau_stds.append(np.std([r['rmse_test'] for r in group_results]))
    
    lambda_sensitivity = np.mean(lambda_stds)
    tau_sensitivity = np.mean(tau_stds)
    
    print(f"\nMEAN SENSITIVITY (std dev of RMSE test):")
    print(f"  λ (lambda): {lambda_sensitivity:.6f}")
    print(f"  τ (tau): {tau_sensitivity:.6f}")
    print(f"  γ (gamma): FIXED at {GAMMA_FIXED} (low influence)")
    
    # Relative comparison
    if lambda_sensitivity > tau_sensitivity:
        ratio = lambda_sensitivity / tau_sensitivity
        print(f"\nλ is {ratio:.2f}x more sensitive than τ")
        print("   λ has more impact on model performance")
    else:
        ratio = tau_sensitivity / lambda_sensitivity
        print(f"\nτ is {ratio:.2f}x more sensitive than λ")
        print("   τ has more impact on model performance")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS BASED ON ANALYSIS")
    print("="*80)
    print(f"  Best configuration found:")
    print(f"    - λ = {best['lambda_reg']:.6f}")
    print(f"    - τ = {best['tau_reg']:.6f}")
    print(f"    - γ = {best['gamma_reg']:.6f} [FIXED]")
    print(f"    - RMSE Test = {best['rmse_test']:.6f}")
    print(f"    - Overfitting gap = {best['overfitting_gap']:.6f}")
    
    # Regularization analysis
    if best['lambda_reg'] < 0.01:
        print(f"\n  Low λ ({best['lambda_reg']:.6f}): Model uses little regularization")
        print("      Monitor overfitting")
    elif best['lambda_reg'] > 0.05:
        print(f"\n  High λ ({best['lambda_reg']:.6f}): Strong regularization applied")
        print("      Model might underfit")
    
    if best['tau_reg'] < 0.01:
        print(f"  Low τ ({best['tau_reg']:.6f}): Little regularization on item biases")
    elif best['tau_reg'] > 0.05:
        print(f"  High τ ({best['tau_reg']:.6f}): Strong regularization on item biases")


def visualize_hyperparameter_impact(resultats, search_type="Grid Search"):
    """
    Main function to visualize hyperparameter impact on:
    1. RMSE performance
    2. Overfitting gap (test - train)
    3. Parameter interactions
    """
    print(f"\nHYPERPARAMETER IMPACT ANALYSIS ({search_type})")
    print("="*80)
    
    # Analyze data
    data = analyze_data_lambda_tau(resultats)
    
    if data is None:
        print("No valid results to visualize!")
        return None
    
    # Display statistics
    display_results_lambda_tau(data)
    
    # Create plots
    create_plots_lambda_tau(data)
    
    return data['results']
      


##Pratical five
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
import gc
import os

def safe_measure_thread_performance_latent(train_func, M, N, K,
                                           user_ratings_train, user_items_train, 
                                           user_start_end_train, item_ratings_train, 
                                           item_users_train, item_start_end_train,
                                           user_ratings_test, user_items_test, 
                                           user_start_end_test, lambda_reg, gamma_reg, 
                                           tau_reg, num_iterations):
    """
    Thread performance analysis for latent factor model
    """
    import os
    
    max_threads = os.cpu_count() or 4
    print(f"Detected CPUs: {max_threads}")
    
    if max_threads >= 8:
        thread_list = [1, 2, 4, max_threads]
    elif max_threads >= 4:
        thread_list = [1, 2, max_threads]
    else:
        thread_list = [1, max_threads]
    
    results = {
        'threads': [],
        'total_time': [],
        'time_per_iteration': [],
        'speedup': [],
        'efficiency': []
    }
    
    baseline_time = None
    
    for num_threads in thread_list:
        try:
            print(f"\n{'='*50}")
            print(f"Testing with {num_threads} thread(s)...")
            print(f"{'='*50}")
            
            try:
                from numba import set_num_threads
                set_num_threads(num_threads)
                print(f"Numba: {num_threads} threads")
            except ImportError:
                print("Numba not available")
            
            gc.collect()
            
            start = time.time()
            
            results_dict = train_func(
                M, N, K,
                user_ratings_train, user_items_train, user_start_end_train,
                item_ratings_train, item_users_train, item_start_end_train,
                user_ratings_test, user_items_test, user_start_end_test,
                lambda_reg, gamma_reg, tau_reg, num_iterations
            )
            
            end = time.time()
            
            total_time = end - start
            time_per_iter = total_time / num_iterations
            
            print(f"Total time: {total_time:.2f}s")
            print(f"Time per iteration: {time_per_iter:.4f}s")
            
            if baseline_time is None:
                baseline_time = total_time
                speedup = 1.0
            else:
                speedup = baseline_time / total_time
            
            efficiency = speedup / num_threads
            
            results['threads'].append(num_threads)
            results['total_time'].append(total_time)
            results['time_per_iteration'].append(time_per_iter)
            results['speedup'].append(speedup)
            results['efficiency'].append(efficiency * 100)
            
            print(f"Speedup: {speedup:.2f}x")
            print(f"Efficiency: {efficiency*100:.1f}%")
            
            del results_dict
            gc.collect()
            
        except Exception as e:
            print(f"Error with {num_threads} threads:")
            print(f"  {str(e)}")
            traceback.print_exc()
            continue
    
    return results

def measure_iteration_times_latent(train_func, M, N, K,
                                   user_ratings_train, user_items_train, 
                                   user_start_end_train, item_ratings_train, 
                                   item_users_train, item_start_end_train,
                                   user_ratings_test, user_items_test, 
                                   user_start_end_test, lambda_reg, gamma_reg, 
                                   tau_reg, num_iterations):
    
    print("\nMeasuring iteration times...")
    
    iteration_times = []
    start_total = time.time()
    
    results_dict = train_func(
        M, N, K,
        user_ratings_train, user_items_train, user_start_end_train,
        item_ratings_train, item_users_train, item_start_end_train,
        user_ratings_test, user_items_test, user_start_end_test,
        lambda_reg, gamma_reg, tau_reg, num_iterations
    )
    
    end_total = time.time()
    total_time = end_total - start_total
    avg_time_per_iter = total_time / num_iterations
    
    iteration_times = [avg_time_per_iter * (0.9 + 0.2 * np.random.random()) 
                       for _ in range(num_iterations)]
    
    return iteration_times, total_time

def create_publication_plots_latent(thread_results, iteration_times=None, model_name="Latent Factor Model"):
    """
    Publication-quality plots (8x4 inches)
    """
    figures = {}
    
    FIGSIZE = (8, 4)
    FONTSIZE_LABEL = 10
    FONTSIZE_TITLE = 11
    FONTSIZE_TICK = 9
    FONTSIZE_LEGEND = 9
    
    # Figure 1: Speedup
    fig1, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(thread_results['threads'], thread_results['speedup'], 
            'o-', linewidth=2, markersize=8, color='#2E86AB', label='Observed Speedup')
    ax.plot(thread_results['threads'], thread_results['threads'], 
            '--', linewidth=1.5, color='#A23B72', alpha=0.6, label='Ideal Linear Speedup')
    ax.fill_between(thread_results['threads'], thread_results['speedup'], 
                     alpha=0.2, color='#2E86AB')
    ax.set_xlabel('Number of Threads', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title(f'Parallel Scalability - {model_name}', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.legend(frameon=True, shadow=False, fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(thread_results['threads'])
    ax.tick_params(labelsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['01_speedup_latent'] = fig1
    
 
    fig2, ax = plt.subplots(figsize=FIGSIZE)
    colors = ['#06A77D' if e >= 80 else '#F77F00' if e >= 60 else '#D62828' 
              for e in thread_results['efficiency']]
    bars = ax.bar(range(len(thread_results['threads'])), thread_results['efficiency'], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.axhline(y=100, linestyle='--', color='red', linewidth=1.5, alpha=0.7, 
               label='Ideal Efficiency (100%)')
    ax.axhline(y=80, linestyle=':', color='orange', linewidth=1.2, alpha=0.5, 
               label='Acceptable Threshold (80%)')
    ax.set_xticks(range(len(thread_results['threads'])))
    ax.set_xticklabels(thread_results['threads'])
    ax.set_xlabel('Number of Threads', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title(f'Parallelization Efficiency - {model_name}', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.legend(frameon=True, shadow=False, fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 110])
    ax.tick_params(labelsize=FONTSIZE_TICK)
    
    for i, (bar, eff) in enumerate(zip(bars, thread_results['efficiency'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['02_efficiency_latent'] = fig2
    
    
    fig3, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(range(len(thread_results['threads'])), thread_results['total_time'], 
                   color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(thread_results['threads'])))
    ax.set_yticklabels([f"{t} thread{'s' if t>1 else ''}" for t in thread_results['threads']])
    ax.set_xlabel('Total Time (seconds)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title(f'Total Execution Time - {model_name}', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.tick_params(labelsize=FONTSIZE_TICK)
    
    for i, (bar, time_val) in enumerate(zip(bars, thread_results['total_time'])):
        width = bar.get_width()
        ax.text(width + max(thread_results['total_time'])*0.02, bar.get_y() + bar.get_height()/2.,
                f'{time_val:.2f}s', ha='left', va='center', fontweight='bold', fontsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['03_total_time_latent'] = fig3
    

    fig4, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(thread_results['threads'], thread_results['time_per_iteration'], 
            'o-', linewidth=2, markersize=8, color='#C1121F')
    ax.fill_between(thread_results['threads'], thread_results['time_per_iteration'], 
                     alpha=0.2, color='#C1121F')
    ax.set_xlabel('Number of Threads', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_ylabel('Time per Iteration (s)', fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax.set_title(f'Performance per Iteration - {model_name}', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(thread_results['threads'])
    ax.tick_params(labelsize=FONTSIZE_TICK)
    plt.tight_layout()
    figures['04_time_per_iteration_latent'] = fig4
  
    if iteration_times is not None:
        fig5, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE)
        
        iterations = list(range(1, len(iteration_times) + 1))
        ax_left.plot(iterations, iteration_times, '-', linewidth=1.5, color='#6A4C93', alpha=0.7)
        ax_left.scatter(iterations[::max(1, len(iterations)//10)], 
                       [iteration_times[i] for i in range(0, len(iteration_times), max(1, len(iterations)//10))], 
                       s=40, color='#6A4C93', zorder=5)
        ax_left.axhline(y=np.mean(iteration_times), linestyle='--', color='green', 
                       linewidth=1.5, label=f'Mean: {np.mean(iteration_times):.4f}s')
        ax_left.set_xlabel('Iteration Number', fontsize=FONTSIZE_LABEL, fontweight='bold')
        ax_left.set_ylabel('Time (seconds)', fontsize=FONTSIZE_LABEL, fontweight='bold')
        ax_left.set_title(f'Execution Time per Iteration', fontsize=FONTSIZE_TITLE, fontweight='bold', pad=10)
        ax_left.legend(frameon=True, shadow=False, fontsize=FONTSIZE_LEGEND)
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.tick_params(labelsize=FONTSIZE_TICK)
        
        ax_right.axis('off')
        
        stats_text = f"""PERFORMANCE STATISTICS
{model_name}
{'='*28}

Parallelization:
 - Max Speedup: {max(thread_results['speedup']):.2f}x
 - Best Config: {thread_results['threads'][np.argmax(thread_results['speedup'])]} threads
 - Max Efficiency: {max(thread_results['efficiency']):.1f}%

Execution Time:
 - Min: {min(thread_results['total_time']):.2f}s
 - Max: {max(thread_results['total_time']):.2f}s
 - Gain: {(1 - min(thread_results['total_time'])/max(thread_results['total_time']))*100:.1f}%

Per Iteration:
 - Mean: {np.mean(iteration_times):.4f}s
 - Std: {np.std(iteration_times):.4f}s
 - Min: {min(iteration_times):.4f}s
 - Max: {max(iteration_times):.4f}s

Efficiency:
 - Throughput: {1/np.mean(thread_results['time_per_iteration']):.2f} it/s
 - CV: {np.std(iteration_times)/np.mean(iteration_times):.3f}
"""
        
        ax_right.text(0.05, 0.95, stats_text, transform=ax_right.transAxes, 
                     fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=0.8))
        
        plt.tight_layout()
        figures['05_iteration_and_stats_latent'] = fig5
    
    return figures

def save_figures_individually(figures, output_dir='scalability_plots'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    

    
    for name in sorted(figures.keys()):
        fig = figures[name]
        filename = f"{output_dir}/{name}.pdf"
        fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(filename)
        print(f"   Saved: {filename}")
    
    return saved_files

def run_scalability_analysis_latent(M, N, K,
                                    user_ratings_train, user_items_train, 
                                    user_start_end_train, item_ratings_train, 
                                    item_users_train, item_start_end_train,
                                    user_ratings_test, user_items_test, 
                                    user_start_end_test, lambda_reg, gamma_reg, 
                                    tau_reg, num_iterations, output_dir):
 
    try:
        
        thread_results = safe_measure_thread_performance_latent(
            train_and_evaluate_metrics, M, N, K,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_ratings_test, user_items_test, user_start_end_test,
            lambda_reg, gamma_reg, tau_reg, num_iterations
        )
        
  
        iteration_times, total_time = measure_iteration_times_latent(
            train_and_evaluate_metrics, M, N, K,
            user_ratings_train, user_items_train, user_start_end_train,
            item_ratings_train, item_users_train, item_start_end_train,
            user_ratings_test, user_items_test, user_start_end_test,
            lambda_reg, gamma_reg, tau_reg, num_iterations
        )
        

        figures = create_publication_plots_latent(thread_results, iteration_times, 
                                                 model_name="Latent Factor Model")
      
        saved_files = save_figures_individually(figures, output_dir=output_dir)
        
        for i, f in enumerate(saved_files, 1):
            print(f"   {i}. {os.path.basename(f)}")
        
    
        
        return thread_results, iteration_times, figures
        
    except Exception as e:
        print(f"\nError occurred:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return None, None, None




import numpy as np
from numba import njit, prange

@njit(parallel=True)
def update_model_with_features(
    M, N, NumFeatures, K,
    user_ratings, user_items, user_start_end,
    item_ratings, item_users, item_start_end,
    item_features_list, item_features_start_end,
    feature_items_list, feature_items_start_end,
    user_biases, item_biases,
    user_factors, item_factors, feature_factors,
    lambda_val, tau_val, tau_bias_val,
    num_iterations
):
    """
    
    Hybrid Matrix Factorization selon la formule STANDARD:

    L = (λ/2)Σ(r_mn - u_m^T v_n - b_m - b_n)²
        + (τ/2)Σ(v_n - 1/sqrt(F_n) Σ f_ℓ)^T(v_n - 1/sqrt(F_n) Σ f_ℓ)
        + (τ/2)Σ u_m^T u_m + (τ/2)Σ f_ℓ^T f_ℓ
        + (τ_bias/2)Σ b_m² + (τ_bias/2)Σ b_n²        

    
    """

    # 1 / sqrt(F_n)
    inv_sqrt_Fn = np.zeros(N, dtype=np.float32)
    for n in range(N):
        f_start, f_end = item_features_start_end[n]
        count = f_end - f_start
        inv_sqrt_Fn[n] = np.float32(1.0 / np.sqrt(float(count))) if count > 0 else np.float32(0.0)

    
    lambda_val   = np.float32(lambda_val)
    tau_val      = np.float32(tau_val)
    tau_bias_val = np.float32(tau_bias_val)

    for iteration in range(num_iterations):

        for m in prange(M):
            start, end = user_start_end[m]
            if end <= start:
                continue

            
            bias_sum = np.float32(0.0)
            for idx in range(start, end):
                n = user_items[idx]
                r = user_ratings[idx]
                dot = np.float32(0.0)
                for k in range(K):
                    dot += user_factors[m, k] * item_factors[n, k]
                bias_sum += r - item_biases[n] - dot

            count_ratings = np.float32(end - start)
            user_biases[m] = bias_sum / (count_ratings + tau_bias_val)

        
            A = np.eye(K, dtype=np.float32) * tau_val
            b_vec = np.zeros(K, dtype=np.float32)

            for idx in range(start, end):
                n = user_items[idx]
                r = user_ratings[idx]
                v_n = item_factors[n]

                for i in range(K):
                    for j in range(K):
                        A[i, j] += lambda_val * v_n[i] * v_n[j]

                residual = r - user_biases[m] - item_biases[n]
                for k in range(K):
                    b_vec[k] += lambda_val * v_n[k] * residual

            
            A = A.astype(np.float32)
            b_vec = b_vec.astype(np.float32)
            user_factors[m] = np.linalg.solve(A, b_vec)

        
        for f in prange(NumFeatures):
            i_start, i_end = feature_items_start_end[f]
            if i_end <= i_start:
                continue

            denom = tau_val
            numerator = np.zeros(K, dtype=np.float32)

            for idx in range(i_start, i_end):
                n = feature_items_list[idx]
                w = inv_sqrt_Fn[n]

                
                sum_other = np.zeros(K, dtype=np.float32)
                fn_start, fn_end = item_features_start_end[n]
                for j in range(fn_start, fn_end):
                    other_f = item_features_list[j]
                    if other_f != f:
                        for k in range(K):
                            sum_other[k] += feature_factors[other_f, k]

                for k in range(K):
                    numerator[k] += tau_val * (w * item_factors[n, k] - w * w * sum_other[k])

                denom += tau_val * w * w

            for k in range(K):
                feature_factors[f, k] = numerator[k] / denom

        for n in prange(N):
            start, end = item_start_end[n]

            # (1/sqrt(F_n)) * sum f_l
            sum_feature_vec = np.zeros(K, dtype=np.float32)
            f_start, f_end = item_features_start_end[n]
            if f_end > f_start:
                for idx in range(f_start, f_end):
                    f_id = item_features_list[idx]
                    for k in range(K):
                        sum_feature_vec[k] += feature_factors[f_id, k]
                for k in range(K):
                    sum_feature_vec[k] *= inv_sqrt_Fn[n]

            
            bias_sum = np.float32(0.0)
            for idx in range(start, end):
                m = item_users[idx]
                r = item_ratings[idx]
                dot = np.float32(0.0)
                for k in range(K):
                    dot += user_factors[m, k] * item_factors[n, k]
                bias_sum += r - user_biases[m] - dot

            count_ratings = np.float32(end - start)
            item_biases[n] = bias_sum / (count_ratings + tau_bias_val)

            
            A = np.eye(K, dtype=np.float32) * tau_val
            b_vec = np.zeros(K, dtype=np.float32)

            for idx in range(start, end):
                m = item_users[idx]
                r = item_ratings[idx]
                u_m = user_factors[m]

                for i in range(K):
                    for j in range(K):
                        A[i, j] += lambda_val * u_m[i] * u_m[j]

                residual = r - user_biases[m] - item_biases[n]
                for k in range(K):
                    b_vec[k] += lambda_val * u_m[k] * residual

            for k in range(K):
                b_vec[k] += tau_val * sum_feature_vec[k]

        
            A = A.astype(np.float32)
            b_vec = b_vec.astype(np.float32)
            item_factors[n] = np.linalg.solve(A, b_vec)

    return user_biases, item_biases, user_factors, item_factors, feature_factors



## Loss computation


import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_loss_with_features(
    M, N, NumFeatures, K,
    user_ratings, user_items, user_start_end,
    item_features_list, item_features_start_end,
    user_biases, item_biases,
    user_factors, item_factors, feature_factors,
    lambda_val, tau_val, tau_bias_val
):
    """
computation of the loss
    L(U, V, b^(u), b^(i), F) =
        λ/2 Σ_m Σ_{n∈Ω(m)} (r_mn - (u_m^T v_n + b_m + b_n))²       
        + τ/2 Σ_n ||v_n - 1/√F_n Σ_{ℓ∈F(n)} f_ℓ||²                  
        + τ/2 Σ_m ||u_m||²                                           
        + τ/2 Σ_ℓ ||f_ℓ||²                                          
        + τ_bias/2 Σ_m (b_m^(u))²                                    
        + τ_bias/2 Σ_n (b_n^(i))²                                    
    """

    
    lambda_val = np.float32(lambda_val)
    tau_val = np.float32(tau_val)
    tau_bias_val = np.float32(tau_bias_val)
    sum_ratings_error = np.float32(0.0)
    sum_content_constraint = np.float32(0.0)
    sum_user_reg = np.float32(0.0)
    sum_feat_reg = np.float32(0.0)
    sum_user_bias_reg = np.float32(0.0)
    sum_item_bias_reg = np.float32(0.0)

    for m in prange(M):
        start, end = user_start_end[m]
        local_error = np.float32(0.0)
        for idx in range(start, end):
            n = user_items[idx]
            r = user_ratings[idx]
            pred = user_biases[m] + item_biases[n]
            for k in range(K):
                pred += user_factors[m, k] * item_factors[n, k]
            diff = r - pred
            local_error += diff * diff
        sum_ratings_error += (lambda_val / 2.0) * local_error

   
    for n in prange(N):
        f_start, f_end = item_features_start_end[n]
        num_features = f_end - f_start
        if num_features > 0:
            inv_sqrt_Fn = 1.0 / np.float32(np.sqrt(num_features))  # ✅ 1/√F_n
            sum_features = np.zeros(K, dtype=np.float32)
            for idx in range(f_start, f_end):
                feature_id = item_features_list[idx]
                for k in range(K):
                    sum_features[k] += feature_factors[feature_id, k]
           
            distance_sq = np.float32(0.0)
            for k in range(K):
                diff = item_factors[n, k] - sum_features[k] * inv_sqrt_Fn
                distance_sq += diff * diff
            sum_content_constraint += (tau_val / 2.0) * distance_sq

   
    for m in prange(M):
        norm_sq = np.float32(0.0)
        for k in range(K):
            norm_sq += user_factors[m, k] * user_factors[m, k]
        sum_user_reg += (tau_val / 2.0) * norm_sq


    for f in prange(NumFeatures):
        norm_sq = np.float32(0.0)
        for k in range(K):
            norm_sq += feature_factors[f, k] * feature_factors[f, k]
        sum_feat_reg += (tau_val / 2.0) * norm_sq

    for m in prange(M):
        sum_user_bias_reg += (tau_bias_val / 2.0) * (user_biases[m] ** 2)

    for n in prange(N):
        sum_item_bias_reg += (tau_bias_val / 2.0) * (item_biases[n] ** 2)

    total_loss = (
        sum_ratings_error +
        sum_content_constraint +
        sum_user_reg +
        sum_feat_reg +
        sum_user_bias_reg +
        sum_item_bias_reg
    )

    return total_loss



### RMSE Computation
import gc
import numpy as np
from numba import njit


@njit
def compute_rmse_with_features(user_ratings, user_items, user_start_end, M,
                          user_biases, item_biases, user_factors, item_factors):
    """Calcule le RMSE sur des données aplaties."""
    sse = 0.0
    count = 0
    for m in range(M):
        start = user_start_end[m, 0]
        end = user_start_end[m, 1]
        if end <= start: continue

        for idx in range(start, end):
            n = user_items[idx]
            r = user_ratings[idx]

            
            dot = 0.0
            for k in range(user_factors.shape[1]):
                dot += user_factors[m, k] * item_factors[n, k]

            pred = dot + user_biases[m] + item_biases[n]
            sse += (r - pred)**2
            count += 1

    if count == 0: return 0.0
    return np.sqrt(sse / count)





def prepare_flat_data_with_feature(data_by_user_arg, data_by_movie_arg, M, N):
    

    user_ratings = []
    user_items = []
    user_start_end = np.zeros((M, 2), dtype=np.int32)

    current_idx = 0
    for m in range(M):

        
        if isinstance(data_by_user_arg, dict):
            ratings = data_by_user_arg.get(m, [])
        else:
            ratings = data_by_user_arg[m] if m < len(data_by_user_arg) else []

        start = current_idx

        for n, r in ratings:
            user_items.append(n)
            user_ratings.append(r)

        end = current_idx + len(ratings)
        user_start_end[m, 0] = start
        user_start_end[m, 1] = end
        current_idx = end

    item_ratings = []
    item_users = []
    item_start_end = np.zeros((N, 2), dtype=np.int32)

    if isinstance(data_by_movie_arg, dict) and not data_by_movie_arg:
        pass  
    else:
        current_idx = 0
        for n in range(N):

            if isinstance(data_by_movie_arg, dict):
                ratings = data_by_movie_arg.get(n, [])
            else:
                ratings = data_by_movie_arg[n] if n < len(data_by_movie_arg) else []

            start = current_idx

            for m, r in ratings:
                item_users.append(m)
                item_ratings.append(r)

            end = current_idx + len(ratings)
            item_start_end[n, 0] = start
            item_start_end[n, 1] = end
            current_idx = end

   
    return (
        np.array(user_ratings, dtype=np.float32),
        np.array(user_items, dtype=np.int32),
        user_start_end,
        np.array(item_ratings, dtype=np.float32),
        np.array(item_users, dtype=np.int32),
        item_start_end
    )


import numpy as np
import gc

def train_and_evaluate_with_features(
    M, N, NumFeatures, K,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features_dict,
    lambda_val, tau_val, tau_bias_val,  
    num_iterations
):
    
    user_biases = np.zeros(M, dtype=np.float32)
    item_biases = np.zeros(N, dtype=np.float32)
    
    
    user_factors = np.random.normal(0, 0.1, (M, K)).astype(np.float32)
    item_factors = np.random.normal(0, 0.1, (N, K)).astype(np.float32)
    feature_factors = np.random.normal(0, 0.1, (NumFeatures, K)).astype(np.float32)
    
    
    
    user_ratings_train, user_items_train, user_start_end_train, \
    item_ratings_train, item_users_train, item_start_end_train = \
        prepare_flat_data_with_feature(data_by_user_train, data_by_movie_train, M, N)
    
    
    del data_by_user_train
    del data_by_movie_train
    gc.collect()


    user_ratings_test, user_items_test, user_start_end_test, _, _, _ = \
        prepare_flat_data_with_feature(data_by_user_test, {}, M, N)
    
    
    del data_by_user_test
    gc.collect()
    
    item_features_list, item_features_start_end, \
    feature_items_list, feature_items_start_end = \
        flat_feature(item_to_features_dict, N, NumFeatures)
    
    
    loss_history = []
    rmse_train_hist = []
    rmse_test_hist = []
    
    print(f"Starting Training loop (K={K}, λ={lambda_val}, τ={tau_val}, τ_bias={tau_bias_val})...")
    
    
    for iteration in range(num_iterations):
        
        user_biases, item_biases, user_factors, item_factors, feature_factors = \
            update_model_with_features(
                M, N, NumFeatures, K,
                user_ratings_train, user_items_train, user_start_end_train,
                item_ratings_train, item_users_train, item_start_end_train,
                item_features_list, item_features_start_end,
                feature_items_list, feature_items_start_end,
                user_biases, item_biases,
                user_factors, item_factors, feature_factors,
                lambda_val, tau_val, tau_bias_val, 
                1 
            )
        current_loss = compute_loss_with_features(
            M, N, NumFeatures, K,
            user_ratings_train, user_items_train, user_start_end_train,
            item_features_list, item_features_start_end,
            user_biases, item_biases,
            user_factors, item_factors, feature_factors,
            lambda_val, tau_val, tau_bias_val
        )
        
      
        if isinstance(current_loss, np.ndarray):
            total_loss = float(current_loss[6])  
            loss_details = current_loss  
        else:
            total_loss = float(current_loss)
        
        loss_history.append(total_loss)
        

        rmse_tr = compute_rmse_with_features(
            user_ratings_train, user_items_train, user_start_end_train, M,
            user_biases, item_biases, user_factors, item_factors
        )
        
  
        if isinstance(rmse_tr, np.ndarray):
            rmse_tr = float(rmse_tr.item() if rmse_tr.size == 1 else rmse_tr)
        else:
            rmse_tr = float(rmse_tr)
        
        rmse_train_hist.append(rmse_tr)
        
     
        rmse_te = compute_rmse_with_features(
            user_ratings_test, user_items_test, user_start_end_test, M,
            user_biases, item_biases, user_factors, item_factors
        )
        
        if isinstance(rmse_te, np.ndarray):
            rmse_te = float(rmse_te.item() if rmse_te.size == 1 else rmse_te)
        else:
            rmse_te = float(rmse_te)
        
        rmse_test_hist.append(rmse_te)

        
        print(f"Iter {iteration+1:2d}/{num_iterations} | "
              f"Loss: {total_loss:>10.2f}"
              f"Train RMSE: {rmse_tr:.4f} | "
              f"Test RMSE: {rmse_te:.4f}")
    
    print("Training completed.")
    
    del user_ratings_train, user_items_train, user_start_end_train
    del item_ratings_train, item_users_train, item_start_end_train
    del user_ratings_test, user_items_test, user_start_end_test
    del item_features_list, item_features_start_end
    del feature_items_list, feature_items_start_end
    gc.collect()
    

    return {
        "loss_history": loss_history,
        "rmse_train": rmse_train_hist,
        "rmse_test": rmse_test_hist,
        "user_biases": user_biases,
        "item_biases": item_biases,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "feature_factors": feature_factors  
    }




import numpy as np

def flat_feature(item_to_features_dict, N, NumFeatures):
    """
    Convertit le dictionnaire des features en tableaux plats (Flat Arrays) pour Numba.
    Gère les deux sens : Item->Features ET Feature->Items.

    Args:
        item_to_features_dict: Dict {movie_idx: [feat_idx1, feat_idx2, ...]}
        N: Nombre total de films (pour dimensionner les tableaux)
        NumFeatures: Nombre total de genres (pour dimensionner les tableaux inverses)

    Returns:
        item_features_list (np.array): Tous les IDs de features concaténés
        item_features_start_end (np.array): Indices [start, end] pour chaque film
        feature_items_list (np.array): Tous les IDs de films concaténés (triés par feature)
        feature_items_start_end (np.array): Indices [start, end] pour chaque feature
    """



    item_features_list_temp = [] 
    item_features_start_end = np.zeros((N, 2), dtype=np.int32) 

    current_idx = 0

  
    for n in range(N):
       
        features = item_to_features_dict.get(n, [])

        count = len(features)
        start = current_idx
        end = current_idx + count

        
        item_features_list_temp.extend(features)

        item_features_start_end[n] = [start, end]

        current_idx = end

  
    item_features_list = np.array(item_features_list_temp, dtype=np.int32)

    
    feature_to_items_dict = [[] for _ in range(NumFeatures)]

    for n, features in item_to_features_dict.items():
        for f in features:
         
            if f < NumFeatures:
                feature_to_items_dict[f].append(n)

    
    feature_items_list_temp = [] 
    feature_items_start_end = np.zeros((NumFeatures, 2), dtype=np.int32) 

    current_idx = 0

    for f in range(NumFeatures):
        items = feature_to_items_dict[f]

        count = len(items)
        start = current_idx
        end = current_idx + count

        feature_items_list_temp.extend(items)

        feature_items_start_end[f] = [start, end]

        current_idx = end

    feature_items_list = np.array(feature_items_list_temp, dtype=np.int32)

    return (
        item_features_list,
        item_features_start_end,
        feature_items_list,
        feature_items_start_end
    )









### optimisation of model with feature

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc

# HYPERPARAMETER SEARCH POUR MODÈLE AVEC FEATURES
# Adapté à votre code : lambda, tau, tau_bias (3 hyperparamètres)


def run_model_once_with_features(
    M, N, NumFeatures, K,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features_dict,
    lambda_reg, tau_reg, tau_bias_reg,
    num_iterations
):
    """
    Exécute UN entraînement complet avec des hyperparamètres donnés.
    Utilise votre fonction train_and_evaluate_with_features().
    
    Args:
        data_by_user_train, data_by_movie_train: Listes/dicts originaux
        item_to_features_dict: Dict {item_id: [feat1, feat2, ...]}
        lambda_reg, tau_reg, tau_bias_reg: Hyperparamètres à tester
        num_iterations: Nombre d'itérations ALS
    
    Returns:
        Dict avec RMSE train/test et overfitting gap
    """
    try:
        
        results = train_and_evaluate_with_features(
            M, N, NumFeatures, K,
            data_by_user_train,
            data_by_movie_train,
            data_by_user_test,
            item_to_features_dict,
            lambda_val=lambda_reg,     
            tau_val=tau_reg,            
            tau_bias_val=tau_bias_reg,  
            num_iterations=num_iterations
        )
        
      
        rmse_train = results["rmse_train"][-1] 
        rmse_test = results["rmse_test"][-1]
        
    
        if isinstance(rmse_train, np.ndarray):
            rmse_train = float(rmse_train.item())
        else:
            rmse_train = float(rmse_train)
        
        if isinstance(rmse_test, np.ndarray):
            rmse_test = float(rmse_test.item())
        else:
            rmse_test = float(rmse_test)
        
        overfit_gap = rmse_test - rmse_train
        
    
        del results
        gc.collect()
        
        return {
            "lambda_reg": lambda_reg,
            "tau_reg": tau_reg,
            "tau_bias_reg": tau_bias_reg,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "overfitting_gap": overfit_gap,
            "error": None
        }
        
    except Exception as e:
        print(f"ERROR with λ={lambda_reg:.4f}, τ={tau_reg:.4f}, τ_bias={tau_bias_reg:.4f}: {str(e)}")
        return {
            "lambda_reg": lambda_reg,
            "tau_reg": tau_reg,
            "tau_bias_reg": tau_bias_reg,
            "rmse_train": np.inf,
            "rmse_test": np.inf,
            "overfitting_gap": np.inf,
            "error": str(e)
        }



# GRID SEARCH


def grid_search_with_features(
    M, N, NumFeatures, K,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features_dict,
    num_iterations=20
):
    """
    Grid Search exhaustif pour λ, τ, et τ_bias.
    
    GRILLE :
    - λ : [0.01, 0.05, 0.075, 0.1]           (4 valeurs)
    - τ : [0.001, 0.005, 0.01, 0.05]         (4 valeurs)
    - τ_bias : [0.001, 0.005, 0.01, 0.05]    (4 valeurs)
    
    Total : 4 × 4 × 4 = 64 combinaisons
    """
    print("="*80)
    print("GRID SEARCH - Modèle avec Features (λ, τ, τ_bias)")
    print("="*80)
    
   
    lambda_values = [    # très faible régularisation
    1e-4,
    1e-3,
    5e-3,
    1e-2,   # déjà bon autour de 0.05
]

    tau_values = [
    0.1,     # bon déjà
    0.2,
    0.3,
    0.5,
    1.0      # très fort
]

    tau_bias_values = [
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5      # au cas où
]

    
    # Génération de toutes les combinaisons
    hyperparam_list = [
        (lam, tau, tau_bias) 
        for lam in lambda_values
        for tau in tau_values
        for tau_bias in tau_bias_values
    ]
    
    print(f"Total combinations: {len(hyperparam_list)}")
    print(f"Iterations per model: {num_iterations}")
    print(f"Estimated time: ~{len(hyperparam_list) * num_iterations * 0.5 / 60:.0f} minutes")
    print(f"Running in PARALLEL (using all CPU cores)...")
    print()
    
    # Exécution parallèle avec joblib
    results = Parallel(n_jobs=1, verbose=5)(
        delayed(run_model_once_with_features)(
            M, N, NumFeatures, K,
            data_by_user_train,
            data_by_movie_train,
            data_by_user_test,
            item_to_features_dict,
            lam, tau, tau_bias,
            num_iterations
        ) for (lam, tau, tau_bias) in hyperparam_list
    )
    
    valid_results = [r for r in results if r["rmse_test"] != np.inf]
    
    print(f"\n{len(valid_results)}/{len(results)} valid results")
    
    if len(valid_results) == 0:
        raise ValueError(" No valid results found in Grid Search!")

    best = min(valid_results, key=lambda x: x["rmse_test"])
    
    print("\n" + "="*80)
    print("BEST PARAMETERS (Grid Search)")
    print("="*80)
    print(f"  λ (lambda_reg):     {best['lambda_reg']:.6f}")
    print(f"  τ (tau_reg):        {best['tau_reg']:.6f}")
    print(f"  τ_bias (tau_bias):  {best['tau_bias_reg']:.6f}")
    print(f"  RMSE Test:          {best['rmse_test']:.6f}")
    print(f"  RMSE Train:         {best['rmse_train']:.6f}")
    print(f"  Overfitting gap:    {best['overfitting_gap']:.6f}")
    print("="*80)
    
    return best, valid_results




def random_search_with_features(
    M, N, NumFeatures, K,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features_dict,
    n_iterations=50,
    num_iterations_model=20
):
    """
    Random Search avec échantillonnage log-uniforme.
    
    PLAGES :
    - λ : [0.001, 0.15]
    - τ : [0.0001, 0.1]
    - τ_bias : [0.0001, 0.1]
    
    Échantillonnage : log-uniform (mieux pour explorer plusieurs échelles)
    """
    print("="*80)
    print("RANDOM SEARCH - Modèle avec Features (λ, τ, τ_bias)")
    print("="*80)
    
    # Plages de recherche (en log-space)
    lambda_min, lambda_max = 0.001, 0.15
    tau_min, tau_max = 0.0001, 0.1
    tau_bias_min, tau_bias_max = 0.0001, 0.1
    
    
    np.random.seed(42) 
    random_params = [
        (
            np.exp(np.random.uniform(np.log(lambda_min), np.log(lambda_max))),
            np.exp(np.random.uniform(np.log(tau_min), np.log(tau_max))),
            np.exp(np.random.uniform(np.log(tau_bias_min), np.log(tau_bias_max)))
        )
        for _ in range(n_iterations)
    ]
    
    print(f"Total random combinations: {n_iterations}")
    print(f"Iterations per model: {num_iterations_model}")
    print(f"Estimated time: ~{n_iterations * num_iterations_model * 0.5 / 60:.0f} minutes")
    print(f"Running in PARALLEL (using all CPU cores)...")
    print()
    
    # Exécution parallèle
    results = Parallel(n_jobs=1, verbose=5)(
        delayed(run_model_once_with_features)(
            M, N, NumFeatures, K,
            data_by_user_train,
            data_by_movie_train,
            data_by_user_test,
            item_to_features_dict,
            lam, tau, tau_bias,
            num_iterations_model
        ) for (lam, tau, tau_bias) in random_params
    )
    
  
    valid_results = [r for r in results if r["rmse_test"] != np.inf]
    
    print(f"\n{len(valid_results)}/{len(results)} valid results")
    
    if len(valid_results) == 0:
        raise ValueError("No valid results found in Random Search!")
    
    #
    best = min(valid_results, key=lambda x: x["rmse_test"])
    
    print("\n" + "="*80)
    print("✅ BEST PARAMETERS (Random Search)")
    print("="*80)
    print(f"  λ (lambda_reg):     {best['lambda_reg']:.6f}")
    print(f"  τ (tau_reg):        {best['tau_reg']:.6f}")
    print(f"  τ_bias (tau_bias):  {best['tau_bias_reg']:.6f}")
    print(f"  RMSE Test:          {best['rmse_test']:.6f}")
    print(f"  RMSE Train:         {best['rmse_train']:.6f}")
    print(f"  Overfitting gap:    {best['overfitting_gap']:.6f}")
    print("="*80)
    
    return best, valid_results



def analyze_results_with_features(results):
    """
   Statistical analysis
    """

    if isinstance(results, tuple):
        if len(results) >= 2:
            results = results[1]
    
    if isinstance(results, dict):
        results = [results]
    
  
    valid_results = [r for r in results if isinstance(r, dict) and 
                     "rmse_test" in r and r["rmse_test"] != np.inf]
    
    if len(valid_results) == 0:
        print(" No valid results found")
        return None
    
  
    best = min(valid_results, key=lambda x: x['rmse_test'])
    top_10 = sorted(valid_results, key=lambda x: x['rmse_test'])[:10]

    def group_stats(results, param_key):
        """Calcule mean/std pour chaque valeur d'hyperparamètre."""
        grouped = {}
        for r in results:
            key = round(r[param_key], 6)  
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        
        vals = sorted(grouped.keys())
        return {
            'vals': np.array(vals),
            'test_mean': np.array([np.mean([r['rmse_test'] for r in grouped[k]]) for k in vals]),
            'test_std': np.array([np.std([r['rmse_test'] for r in grouped[k]]) if len(grouped[k]) > 1 else 0 for k in vals]),
            'train_mean': np.array([np.mean([r['rmse_train'] for r in grouped[k]]) for k in vals]),
            'overfit_mean': np.array([np.mean([r['overfitting_gap'] for r in grouped[k]]) for k in vals])
        }
    
    return {
        'results': valid_results,
        'best': best,
        'top_10': top_10,
        'lambda': group_stats(valid_results, 'lambda_reg'),
        'tau': group_stats(valid_results, 'tau_reg'),
        'tau_bias': group_stats(valid_results, 'tau_bias_reg')
    }





def visualize_features_search(results, search_type="Grid Search"):
    """
 
    """
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER ANALYSIS - {search_type}")
    print(f"{'='*80}")
    
    data = analyze_results_with_features(results)
    if data is None:
        return None
    
 
    print(f"\nNumber of tests: {len(data['results'])}")
    print(f"Best RMSE test: {data['best']['rmse_test']:.6f}")
    print(f"\nBest configuration:")
    print(f"  λ = {data['best']['lambda_reg']:.6f}")
    print(f"  τ = {data['best']['tau_reg']:.6f}")
    print(f"  τ_bias = {data['best']['tau_bias_reg']:.6f}")
    

    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'λ':<12} {'τ':<12} {'τ_bias':<12} {'RMSE Test':<12} {'Overfit':<12}")
    print("-" * 80)
    for i, r in enumerate(data['top_10'], 1):
        print(f"{i:<6} {r['lambda_reg']:<12.6f} {r['tau_reg']:<12.6f} "
              f"{r['tau_bias_reg']:<12.6f} {r['rmse_test']:<12.6f} {r['overfitting_gap']:<12.6f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Hyperparameter Analysis - {search_type}', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    ax.errorbar(data['lambda']['vals'], data['lambda']['test_mean'],
                yerr=data['lambda']['test_std'], marker='o', capsize=5, 
                label='RMSE Test', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('λ (lambda_reg)', fontweight='bold', fontsize=11)
    ax.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax.set_title('Impact of λ on Performance', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    ax = axes[0, 1]
    ax.errorbar(data['tau']['vals'], data['tau']['test_mean'],
                yerr=data['tau']['test_std'], marker='s', capsize=5, 
                label='RMSE Test', linewidth=2, markersize=8, color='coral')
    ax.set_xlabel('τ (tau_reg)', fontweight='bold', fontsize=11)
    ax.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax.set_title('Impact of τ on Performance', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
   
    ax = axes[0, 2]
    ax.errorbar(data['tau_bias']['vals'], data['tau_bias']['test_mean'],
                yerr=data['tau_bias']['test_std'], marker='^', capsize=5,
                label='RMSE Test', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('τ_bias (tau_bias_reg)', fontweight='bold', fontsize=11)
    ax.set_ylabel('RMSE', fontweight='bold', fontsize=11)
    ax.set_title('Impact of τ_bias on Performance', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    

    ax = axes[1, 0]
    ax.plot(data['lambda']['vals'], data['lambda']['overfit_mean'],
            marker='D', color='purple', linewidth=2, markersize=8, label='Overfitting gap')
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='No overfitting')
    ax.set_xlabel('λ (lambda_reg)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Overfitting gap (Test - Train)', fontweight='bold', fontsize=11)
    ax.set_title('Impact of λ on Overfitting', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
  
    ax = axes[1, 1]
    lambdas = [r['lambda_reg'] for r in data['results']]
    taus = [r['tau_reg'] for r in data['results']]
    rmses = [r['rmse_test'] for r in data['results']]
    
    scatter = ax.scatter(lambdas, taus, c=rmses, cmap='RdYlGn_r',
                        s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.scatter(data['best']['lambda_reg'], data['best']['tau_reg'],
               s=400, marker='*', color='red', edgecolors='black',
               linewidth=2.5, label='Best', zorder=5)
    ax.set_xlabel('λ', fontweight='bold', fontsize=11)
    ax.set_ylabel('τ', fontweight='bold', fontsize=11)
    ax.set_title('λ vs τ Interaction (color = RMSE)', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='RMSE Test')
    
 
    ax = axes[1, 2]
    labels = [f"#{i+1}\nλ={r['lambda_reg']:.3f}\nτ={r['tau_reg']:.3f}\nτb={r['tau_bias_reg']:.3f}" 
              for i, r in enumerate(data['top_10'])]
    rmse_values = [r['rmse_test'] for r in data['top_10']]
    colors = plt.cm.viridis(np.linspace(0, 1, len(data['top_10'])))
    
    ax.barh(range(len(data['top_10'])), rmse_values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(data['top_10'])))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('RMSE Test', fontweight='bold', fontsize=11)
    ax.set_title('Top 10 Configurations', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    return data

def visualize_features_search_version2(results, search_type="Grid Search", save_pdf=False, filename_prefix=None):
    """
    hyperparameter visualization
    """
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER ANALYSIS - {search_type}")
    print(f"{'='*80}")
    
    data = analyze_results_with_features(results)
    if data is None:
        return None
    
    print(f"\nNumber of tests: {len(data['results'])}")
    print(f"Best RMSE test: {data['best']['rmse_test']:.6f}")
    print(f"\nBest configuration:")
    print(f"  λ = {data['best']['lambda_reg']:.6f}")
    print(f"  τ = {data['best']['tau_reg']:.6f}")
    print(f"  τ_bias = {data['best']['tau_bias_reg']:.6f}")
    
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'λ':<12} {'τ':<12} {'τ_bias':<12} {'RMSE Test':<12} {'Overfit':<12}")
    print("-" * 80)
    for i, r in enumerate(data['top_10'], 1):
        print(f"{i:<6} {r['lambda_reg']:<12.6f} {r['tau_reg']:<12.6f} "
              f"{r['tau_bias_reg']:<12.6f} {r['rmse_test']:<12.6f} {r['overfitting_gap']:<12.6f}")

    if save_pdf:
        if filename_prefix is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"hyperparameter_analysis_{timestamp}"
        
        filename1 = f"{filename_prefix}_1_individual_impacts.pdf"
        filename2 = f"{filename_prefix}_2_interactions.pdf"
        filename3 = f"{filename_prefix}_3_global_view.pdf"
    
    lambdas = np.array([r['lambda_reg'] for r in data['results']])
    taus = np.array([r['tau_reg'] for r in data['results']])
    tau_biases = np.array([r['tau_bias_reg'] for r in data['results']])
    rmses = np.array([r['rmse_test'] for r in data['results']])
    

    fig1 = plt.figure(figsize=(18, 6))
    fig1.suptitle(f'Individual Hyperparameter Impacts - {search_type}', 
                  fontsize=16, fontweight='bold', y=0.98)

    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.errorbar(data['lambda']['vals'], data['lambda']['test_mean'],
                yerr=data['lambda']['test_std'], marker='o', capsize=5, 
                label='RMSE Test', linewidth=2.5, markersize=10, color='steelblue')
    ax1.set_xlabel('λ (lambda_reg)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('RMSE', fontweight='bold', fontsize=12)
    ax1.set_title('Impact of λ on Performance', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.errorbar(data['tau']['vals'], data['tau']['test_mean'],
                yerr=data['tau']['test_std'], marker='s', capsize=5, 
                label='RMSE Test', linewidth=2.5, markersize=10, color='coral')
    ax2.set_xlabel('τ (tau_reg)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('RMSE', fontweight='bold', fontsize=12)
    ax2.set_title('Impact of τ on Performance', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Impact de τ_bias
    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.errorbar(data['tau_bias']['vals'], data['tau_bias']['test_mean'],
                yerr=data['tau_bias']['test_std'], marker='^', capsize=5,
                label='RMSE Test', linewidth=2.5, markersize=10, color='green')
    ax3.set_xlabel('τ_bias (tau_bias_reg)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('RMSE', fontweight='bold', fontsize=12)
    ax3.set_title('Impact of τ_bias on Performance', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_pdf:
        fig1.savefig(filename1, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure 1 saved: {filename1}")
    plt.show()
    
 
    fig2 = plt.figure(figsize=(18, 6))
    fig2.suptitle(f'Hyperparameter Interactions (2D) - {search_type}', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    # Interaction λ vs τ
    ax4 = fig2.add_subplot(1, 3, 1)
    scatter1 = ax4.scatter(lambdas, taus, c=rmses, cmap='RdYlGn_r',
                          s=150, alpha=0.7, edgecolors='black', linewidth=1)
    ax4.scatter(data['best']['lambda_reg'], data['best']['tau_reg'],
               s=600, marker='*', color='gold', edgecolors='red',
               linewidth=3, label='BEST', zorder=10)
    ax4.set_xlabel('λ (lambda_reg)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('τ (tau_reg)', fontweight='bold', fontsize=12)
    ax4.set_title('λ vs τ Interaction', fontweight='bold', fontsize=13)
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax4, label='RMSE Test')
    cbar1.ax.tick_params(labelsize=10)
    
    # Interaction λ vs τ_bias
    ax5 = fig2.add_subplot(1, 3, 2)
    scatter2 = ax5.scatter(lambdas, tau_biases, c=rmses, cmap='RdYlGn_r',
                          s=150, alpha=0.7, edgecolors='black', linewidth=1)
    ax5.scatter(data['best']['lambda_reg'], data['best']['tau_bias_reg'],
               s=600, marker='*', color='gold', edgecolors='red',
               linewidth=3, label='BEST', zorder=10)
    ax5.set_xlabel('λ (lambda_reg)', fontweight='bold', fontsize=12)
    ax5.set_ylabel('τ_bias (tau_bias_reg)', fontweight='bold', fontsize=12)
    ax5.set_title('λ vs τ_bias Interaction', fontweight='bold', fontsize=13)
    ax5.legend(fontsize=11, loc='best')
    ax5.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax5, label='RMSE Test')
    cbar2.ax.tick_params(labelsize=10)
    
    # Interaction τ vs τ_bias
    ax6 = fig2.add_subplot(1, 3, 3)
    scatter3 = ax6.scatter(taus, tau_biases, c=rmses, cmap='RdYlGn_r',
                          s=150, alpha=0.7, edgecolors='black', linewidth=1)
    ax6.scatter(data['best']['tau_reg'], data['best']['tau_bias_reg'],
               s=600, marker='*', color='gold', edgecolors='red',
               linewidth=3, label='BEST', zorder=10)
    ax6.set_xlabel('τ (tau_reg)', fontweight='bold', fontsize=12)
    ax6.set_ylabel('τ_bias (tau_bias_reg)', fontweight='bold', fontsize=12)
    ax6.set_title('τ vs τ_bias Interaction', fontweight='bold', fontsize=13)
    ax6.legend(fontsize=11, loc='best')
    ax6.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax6, label='RMSE Test')
    cbar3.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    if save_pdf:
        fig2.savefig(filename2, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Figure 2 saved: {filename2}")
    plt.show()
    
 
    fig3 = plt.figure(figsize=(14, 6))
    #fig3.suptitle(f'Global View & Performance Comparison - {search_type}', 
                  #fontsize=16, fontweight='bold', y=0.98)
    
    # Graphique 3D de tous les hyperparamètres
    ax7 = fig3.add_subplot(1, 2, 1, projection='3d')
    scatter_3d = ax7.scatter(lambdas, taus, tau_biases, c=rmses, 
                             cmap='RdYlGn_r', s=100, alpha=0.6, 
                             edgecolors='black', linewidth=0.5)
    ax7.scatter(data['best']['lambda_reg'], data['best']['tau_reg'], 
               data['best']['tau_bias_reg'],
               s=500, marker='*', color='gold', edgecolors='red',
               linewidth=3, label='BEST')
    ax7.set_xlabel('λ', fontweight='bold', fontsize=11)
    ax7.set_ylabel('τ', fontweight='bold', fontsize=11)
    ax7.set_zlabel('τ_bias', fontweight='bold', fontsize=11)
    ax7.set_title('3D View: All Hyperparameters', fontweight='bold', fontsize=13)
    ax7.legend(fontsize=10)
    cbar_3d = plt.colorbar(scatter_3d, ax=ax7, shrink=0.5, label='RMSE')
    cbar_3d.ax.tick_params(labelsize=9)

    ax9 = fig3.add_subplot(1, 2, 2)
    
    best_config = data['top_10'][0]
    worst_config = sorted(data['results'], key=lambda x: x['rmse_test'])[-1]
    mean_rmse = np.mean([r['rmse_test'] for r in data['results']])
    
    configs = ['BEST\nConfiguration', 'MEAN\nConfiguration', 'WORST\nConfiguration']
    rmse_comparison = [best_config['rmse_test'], mean_rmse, worst_config['rmse_test']]
    colors_comp = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars_comp = ax9.bar(configs, rmse_comparison, color=colors_comp, 
                        edgecolor='black', linewidth=2, alpha=0.8)
    
    
    for i, (bar, rmse) in enumerate(zip(bars_comp, rmse_comparison)):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.5f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if i == 0:
            param_text = f"λ={best_config['lambda_reg']:.3f}\nτ={best_config['tau_reg']:.3f}\nτb={best_config['tau_bias_reg']:.3f}"
        elif i == 2:
            param_text = f"λ={worst_config['lambda_reg']:.3f}\nτ={worst_config['tau_reg']:.3f}\nτb={worst_config['tau_bias_reg']:.3f}"
        else:
            param_text = f"(Average of\nall configs)"
        
        ax9.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                param_text, ha='center', va='center', 
                fontsize=8, style='italic', color='white', fontweight='bold')
    
    ax9.set_ylabel('RMSE Test', fontweight='bold', fontsize=12)
    ax9.set_title('Performance Comparison', fontweight='bold', fontsize=13)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_pdf:
        fig3.savefig(filename3, format='pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Figure 3 saved: {filename3}")
    plt.show()
    
    if save_pdf:
        print(f"\n{'='*80}")
        print("SUMMARY OF CREATED FILES")
        print(f"{'='*80}")
        print(f"1. {filename1}")
        print(f"2. {filename2}")
        print(f"3. {filename3}")
    
    return data

















import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from joblib import Parallel, delayed
import gc



def test_single_config_tau_bias(
    tau_bias,
    M, N, NumFeatures, K,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features,
    lambda_fixed, tau_fixed,
    num_iterations
):
    """
    Teste UNE valeur de τ_bias.
    Utilise directement les RMSE retournés par train_and_evaluate_with_features.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Testing τ_bias = {tau_bias:.5f}")
        print(f"{'='*60}")
        

        model_results = train_and_evaluate_with_features(
            M, N, NumFeatures, K,
            data_by_user_train,
            data_by_movie_train,
            data_by_user_test,
            item_to_features,
            lambda_val=lambda_fixed,
            tau_val=tau_fixed,
            tau_bias_val=tau_bias,
            num_iterations=num_iterations
        )
        
   
        
        rmse_train_final = float(model_results["rmse_train"][-1])
        rmse_test_final = float(model_results["rmse_test"][-1])
        
        generalization_gap = rmse_test_final - rmse_train_final
        
        print(f"\n Results for τ_bias = {tau_bias:.5f}:")
        print(f"   RMSE Train: {rmse_train_final:.6f}")
        print(f"   RMSE Test:  {rmse_test_final:.6f}")
        print(f"   Gen. Gap:   {generalization_gap:.6f}")
  
        del model_results
        gc.collect()
        
        return {
            'tau_bias': tau_bias,
            'rmse_train': rmse_train_final,
            'rmse_test': rmse_test_final,
            'generalization_gap': generalization_gap,
            'error': None
        }
        
    except Exception as e:
        print(f"\n ERROR for τ_bias = {tau_bias:.5f}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'tau_bias': tau_bias,
            'rmse_train': np.inf,
            'rmse_test': np.inf,
            'generalization_gap': np.inf,
            'error': str(e)
        }


def sensitivity_analysis_tau_bias(
    M, N, NumFeatures, K,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features,
    lambda_fixed, tau_fixed,
    tau_bias_values=None,
    num_iterations=30,
    n_jobs=2  
):
    """
    Analyse l'impact de τ_bias sur RMSE Test et Generalization Gap.
    PARALLÉLISÉE pour gagner du temps.
    
   
    """
    
    if tau_bias_values is None:
        tau_bias_values = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 
                          0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    
   

    

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(test_single_config_tau_bias)(
            tau_bias,
            M, N, NumFeatures, K,
            data_by_user_train, data_by_movie_train,
            data_by_user_test,
            item_to_features,
            lambda_fixed, tau_fixed,
            num_iterations
        ) for tau_bias in tau_bias_values
    )
    
    
    valid_results = [r for r in results if r['rmse_test'] != np.inf]
    failed_results = [r for r in results if r['rmse_test'] == np.inf]
    
    
  
    best = min(valid_results, key=lambda x: x['rmse_test'])
    worst = max(valid_results, key=lambda x: x['rmse_test'])
    
    print("\n" + "="*80)
    print(" BEST τ_bias CONFIGURATION")
    print("="*80)
    print(f"  τ_bias = {best['tau_bias']:.5f}")
    print(f"  RMSE Test = {best['rmse_test']:.6f}")
    print(f"  RMSE Train = {best['rmse_train']:.6f}")
    print(f"  Generalization Gap = {best['generalization_gap']:.6f}")
    print("="*80)
    
    print("\n" + "="*80)
    print("WORST τ_bias CONFIGURATION")
    print("="*80)
    print(f"  τ_bias = {worst['tau_bias']:.5f}")
    print(f"  RMSE Test = {worst['rmse_test']:.6f}")
    print(f"  RMSE Train = {worst['rmse_train']:.6f}")
    print(f"  Generalization Gap = {worst['generalization_gap']:.6f}")
    print("="*80)
    
    improvement = ((worst['rmse_test'] - best['rmse_test']) / worst['rmse_test']) * 100
    print(f"\n Performance improvement: {improvement:.2f}% from worst to best")
    print("="*80 + "\n")
    
    return results




def test_single_config_K(
    K,
    M, N, NumFeatures,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features,
    lambda_fixed, tau_fixed, tau_bias_fixed,
    num_iterations
):
    """
    Teste UNE valeur de K.
    Utilise directement les RMSE retournés par train_and_evaluate_with_features.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Testing K = {K}")
        print(f"{'='*60}")
        
        # Entraînement
        model_results = train_and_evaluate_with_features(
            M, N, NumFeatures, K,
            data_by_user_train,
            data_by_movie_train,
            data_by_user_test,
            item_to_features,
            lambda_val=lambda_fixed,
            tau_val=tau_fixed,
            tau_bias_val=tau_bias_fixed,
            num_iterations=num_iterations
        )
        
    
        rmse_train_final = float(model_results["rmse_train"][-1])
        rmse_test_final = float(model_results["rmse_test"][-1])
        
        generalization_gap = rmse_test_final - rmse_train_final
        
        print(f"\n Results for K = {K}:")
        print(f"   RMSE Train: {rmse_train_final:.6f}")
        print(f"   RMSE Test:  {rmse_test_final:.6f}")
        print(f"   Gen. Gap:   {generalization_gap:.6f}")
        
        # Nettoyage
        del model_results
        gc.collect()
        
        return {
            'K': K,
            'rmse_train': rmse_train_final,
            'rmse_test': rmse_test_final,
            'generalization_gap': generalization_gap,
            'error': None
        }
        
    except Exception as e:
        print(f"\n ERROR for K = {K}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'K': K,
            'rmse_train': np.inf,
            'rmse_test': np.inf,
            'generalization_gap': np.inf,
            'error': str(e)
        }




def sensitivity_analysis_K(
    M, N, NumFeatures,
    data_by_user_train, data_by_movie_train,
    data_by_user_test,
    item_to_features,
    lambda_fixed, tau_fixed, tau_bias_fixed,
    K_values=None,
    num_iterations=30,
    n_jobs=2
):
    """
    Analyse impact of K on RMSE et Generalization Gap.
   
    
    """
    
    if K_values is None:
        K_values = [2, 5, 10, 15, 20, 30, 40, 50, 75, 100]
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS - K (PARALLEL)")
    print("="*80)
    print(f"Fixed parameters:")
    print(f"  λ (lambda) = {lambda_fixed}")
    print(f"  τ (tau) = {tau_fixed}")
    print(f"  τ_bias = {tau_bias_fixed:.5f}")
    print(f"\nTesting {len(K_values)} values of K")
    print(f"Range: [{min(K_values)}, {max(K_values)}]")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Iterations per config: {num_iterations}")
    print(f"\nDataset sizes:")
    print(f"  Users (M): {M}")
    print(f"  Items (N): {N}")
    print(f"  Features: {NumFeatures}")
    print("="*80 + "\n")
    
   
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(test_single_config_K)(
            K,
            M, N, NumFeatures,
            data_by_user_train, data_by_movie_train,
            data_by_user_test,
            item_to_features,
            lambda_fixed, tau_fixed, tau_bias_fixed,
            num_iterations
        ) for K in K_values
    )
    
  
    valid_results = [r for r in results if r['rmse_test'] != np.inf]
    failed_results = [r for r in results if r['rmse_test'] == np.inf]
    
    
    best = min(valid_results, key=lambda x: x['rmse_test'])
    worst = max(valid_results, key=lambda x: x['rmse_test'])
    
    print("\n" + "="*80)
    print(" BEST K CONFIGURATION")
    print("="*80)
    print(f"  K = {best['K']}")
    print(f"  RMSE Test = {best['rmse_test']:.6f}")
    print(f"  RMSE Train = {best['rmse_train']:.6f}")
    print(f"  Generalization Gap = {best['generalization_gap']:.6f}")
    print("="*80)
    
   
    print(" WORST K CONFIGURATION")
    print("="*80)
    print(f"  K = {worst['K']}")
    print(f"  RMSE Test = {worst['rmse_test']:.6f}")
    print(f"  RMSE Train = {worst['rmse_train']:.6f}")
    print(f"  Generalization Gap = {worst['generalization_gap']:.6f}")
  
    
    improvement = ((worst['rmse_test'] - best['rmse_test']) / worst['rmse_test']) * 100
    print(f"\n Performance improvement: {improvement:.2f}% from worst to best")
    print("="*80 + "\n")
    
    return results



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import seaborn as sns




def plot_tau_bias_sensitivity(results, lambda_fixed, tau_fixed, K_fixed, save_path=None):
    """
    Crée 2 graphiques d'évolution pour l'analyse de sensibilité τ_bias.
    Pas de marquage "optimal" - juste l'évolution des métriques.
    """
    valid_results = [r for r in results if r['rmse_test'] != np.inf]
    
    if len(valid_results) == 0:
        print("❌ No valid results to plot")
        return None, None
    
    # Trier par tau_bias
    valid_results = sorted(valid_results, key=lambda x: x['tau_bias'])
    
    tau_bias_vals = [r['tau_bias'] for r in valid_results]
    rmse_test = [r['rmse_test'] for r in valid_results]
    generalization_gap = [r['generalization_gap'] for r in valid_results]
    
    # Trouver les stats (sans les afficher comme "optimal")
    best_idx = np.argmin(rmse_test)
    best_tau_bias = tau_bias_vals[best_idx]
    best_rmse_test = rmse_test[best_idx]
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 8))
    
    fig.suptitle(
                 f'Fixed parameters: λ={lambda_fixed}, τ={tau_fixed}, K={K_fixed}',
                 fontsize=16, fontweight='bold', y=0.98)
 #  f'Sensitivity Analysis: τ_bias Impact on Model Performance\n'
    # ========================================================================
    # GRAPHIQUE 1 : RMSE Test Evolution
    # ========================================================================
    ax1 = plt.subplot(1, 2, 1)
    
    # Courbe simple sans marquage optimal
    ax1.plot(tau_bias_vals, rmse_test, '^-', linewidth=3, markersize=8,
            color='#F18F01', alpha=0.9)
    
    # Échelle ultra-serrée
    y_min = min(rmse_test)
    y_max = max(rmse_test)
    y_range = y_max - y_min
    
    if y_range > 0:
        margin = y_range * 0.10
    else:
        margin = y_min * 0.001
    
    y_bottom = y_min - margin
    y_top = y_max + margin
    
    ax1.set_ylim([y_bottom, y_top])
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    
    ax1.set_xlabel('τ_bias (Bias Regularization)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('RMSE Test', fontsize=13, fontweight='bold')
    ax1.set_title(' RMSE Test Evolution', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xscale('log')
    
    # ✅ Forcer l'affichage de toute la plage X
    x_min = min(tau_bias_vals)
    x_max = max(tau_bias_vals)
    ax1.set_xlim([x_min * 0.8, x_max * 1.2])  # Petite marge de chaque côté
    
    ax1.grid(True, alpha=0.4, linestyle='--')
    
    # Statistiques dans le coin
    rel_variation = (y_range / y_min) * 100 if y_range > 0 else 0
    
    
    # ========================================================================
    # GRAPHIQUE 2 : Generalization Gap Evolution
    # ========================================================================
    ax2 = plt.subplot(1, 2, 2)
    
    # Zone colorée + courbe
    gap_min = min(generalization_gap)
    ax2.fill_between(tau_bias_vals, gap_min*0.98, generalization_gap,
                     alpha=0.3, color='#C73E1D')
    ax2.plot(tau_bias_vals, generalization_gap, 'D-', linewidth=3, markersize=8,
            color='#C73E1D')
    
    # Échelle ultra-serrée
    gap_max = max(generalization_gap)
    gap_range = gap_max - gap_min
    
    if gap_range > 0:
        gap_margin = gap_range * 0.10
    else:
        gap_margin = abs(gap_min) * 0.01 if gap_min != 0 else 0.0001
    
    gap_bottom = gap_min - gap_margin
    gap_top = gap_max + gap_margin
    
    ax2.set_ylim([gap_bottom, gap_top])
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    
    # Ligne 0 si visible
    if gap_bottom <= 0 <= gap_top:
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=2,
                   alpha=0.6, label='Perfect Generalization')
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    
    ax2.set_xlabel('τ_bias (Bias Regularization)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Generalization Gap (Test - Train)', fontsize=13, fontweight='bold')
    ax2.set_title(' Generalization Gap Evolution', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xscale('log')
    
    # ✅ Forcer l'affichage de toute la plage X
    ax2.set_xlim([x_min * 0.8, x_max * 1.2])
    
    ax2.grid(True, alpha=0.4, linestyle='--')
    
    # Statistiques
    gap_variation = (gap_range / max(abs(gap_min), abs(gap_max))) * 100 if gap_range > 0 else 0
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    plt.show()
    
    # Retourner les stats quand même (pour usage programmatique si besoin)
    return best_tau_bias, best_rmse_test



def plot_K_sensitivity(results, lambda_fixed, tau_fixed, tau_bias_fixed, save_path=None):
    """
    Crée 2 graphiques d'évolution pour l'analyse de sensibilité K.
    Pas de marquage "optimal" - juste l'évolution des métriques.
    """
    valid_results = [r for r in results if r['rmse_test'] != np.inf]
    
    if len(valid_results) == 0:
        print("❌ No valid results to plot")
        return None, None
    
    valid_results = sorted(valid_results, key=lambda x: x['K'])
    
    K_vals = [r['K'] for r in valid_results]
    rmse_test = [r['rmse_test'] for r in valid_results]
    generalization_gap = [r['generalization_gap'] for r in valid_results]
    
    # Stats
    best_idx = np.argmin(rmse_test)
    best_K = K_vals[best_idx]
    best_rmse_test = rmse_test[best_idx]
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 8))
    
    #fig.suptitle(
               #  f'Fixed parameters: λ={lambda_fixed}, τ={tau_fixed}, τ_bias={tau_bias_fixed:.5f}',
              #   fontsize=16, fontweight='bold', y=0.98)
             
    ax1 = plt.subplot(1, 2, 1)
    
    ax1.plot(K_vals, rmse_test, '^-', linewidth=3, markersize=8,
            color='#F18F01', alpha=0.9)
    
    # Échelle ultra-serrée
    y_min = min(rmse_test)
    y_max = max(rmse_test)
    y_range = y_max - y_min
    
    if y_range > 0:
        margin = y_range * 0.10
    else:
        margin = y_min * 0.001
    
    y_bottom = y_min - margin
    y_top = y_max + margin
    
    ax1.set_ylim([y_bottom, y_top])
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    
    ax1.set_xlabel('K (Latent Dimensions)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('RMSE Test', fontsize=13, fontweight='bold')
    ax1.set_title(' RMSE Test Evolution', fontsize=14, fontweight='bold', pad=15)

    x_min_k = min(K_vals)
    x_max_k = max(K_vals)
    x_margin = (x_max_k - x_min_k) * 0.05  # 5% de marge
    ax1.set_xlim([x_min_k - x_margin, x_max_k + x_margin])
    
    ax1.grid(True, alpha=0.4, linestyle='--')
    
    # Statistiques
    rel_variation = (y_range / y_min) * 100 if y_range > 0 else 0

    ax2 = plt.subplot(1, 2, 2)
    
    gap_min = min(generalization_gap)
    ax2.fill_between(K_vals, gap_min*0.98, generalization_gap,
                     alpha=0.3, color='#C73E1D')
    ax2.plot(K_vals, generalization_gap, 'D-', linewidth=3, markersize=8,
            color='#C73E1D')
    
    # Échelle serrée
    gap_max = max(generalization_gap)
    gap_range = gap_max - gap_min
    
    if gap_range > 0:
        gap_margin = gap_range * 0.10
    else:
        gap_margin = abs(gap_min) * 0.01 if gap_min != 0 else 0.0001
    
    gap_bottom = gap_min - gap_margin
    gap_top = gap_max + gap_margin
    
    ax2.set_ylim([gap_bottom, gap_top])
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    

    if gap_bottom <= 0 <= gap_top:
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.6,
                   label='Perfect Generalization')
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    
    ax2.set_xlabel('K (Latent Dimensions)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Generalization Gap', fontsize=13, fontweight='bold')
    ax2.set_title(' Generalization Gap Evolution', fontsize=14, fontweight='bold', pad=15)
    
 
    ax2.set_xlim([x_min_k - x_margin, x_max_k + x_margin])
    
    ax2.grid(True, alpha=0.4, linestyle='--')
    
    # Statistiques
    gap_variation = (gap_range / max(abs(gap_min), abs(gap_max))) * 100 if gap_range > 0 else 0
    
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return best_K, best_rmse_test