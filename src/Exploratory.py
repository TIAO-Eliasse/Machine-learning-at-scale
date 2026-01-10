import csv
import random
import matplotlib.pyplot as plt
import numpy as np 

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

## Fubction to create the structure of data of users and data of movies

def build_user_movie_index(ratings_file):
    """
    Load the ratings file and create structures to access data by user and by movie.
    """
    user_to_index = {}
    movie_to_index = {}
    index_to_user = []
    index_to_movie = []

    with open(ratings_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        user_count = 0
        movie_count = 0

        for userId, movieId, rating_str, timestamp in reader:
            if userId not in user_to_index:
                user_to_index[userId] = user_count
                index_to_user.append(userId)
                user_count += 1
            if movieId not in movie_to_index:
                movie_to_index[movieId] = movie_count
                index_to_movie.append(movieId)
                movie_count += 1

    n_users = len(index_to_user)
    n_movies = len(index_to_movie)
    data_by_user = [[] for _ in range(n_users)]
    data_by_movie = [[] for _ in range(n_movies)]

    with open(ratings_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for userId, movieId, rating_str, timestamp in reader:
            u = user_to_index[userId]
            m = movie_to_index[movieId]
            r = float(rating_str)
            data_by_user[u].append((m, r))
            data_by_movie[m].append((u, r))

    return data_by_user, data_by_movie, index_to_user, index_to_movie,user_to_index,movie_to_index


# Function to split the data


def split_dataset(data_by_user, n_movies, train_ratio=0.9):
    """
   
    """

    n_users = len(data_by_user)


    data_by_user_train = [[] for _ in range(n_users)]
    data_by_user_test  = [[] for _ in range(n_users)]
    data_by_movie_train = [[] for _ in range(n_movies)]
    data_by_movie_test  = [[] for _ in range(n_movies)]

   
    for user_idx, ratings in enumerate(data_by_user):

        ratings_copy = ratings[:]  
        random.shuffle(ratings_copy)

        split_point = int(train_ratio * len(ratings_copy))

        for i, (movie_idx, rating) in enumerate(ratings_copy):
            if i < split_point:
                data_by_user_train[user_idx].append((movie_idx, rating))
                data_by_movie_train[movie_idx].append((user_idx, rating))
            else:
                data_by_user_test[user_idx].append((movie_idx, rating))
                data_by_movie_test[movie_idx].append((user_idx, rating))

    return (data_by_user_train, data_by_user_test,
            data_by_movie_train, data_by_movie_test)




class RatingsAnalysis:
    def __init__(self, train_values, test_values, show=True):
        self.train_values = np.array(train_values)
        self.test_values = np.array(test_values)
        self.show = show

        # Calcul automatique
        self.train_counts = Counter(train_values)
        self.test_counts = Counter(test_values)
        self.ratings = sorted(set(train_values) | set(test_values))
        self.n_train = len(train_values)
        self.n_test = len(test_values)

        self.ratings_per_user_train = [1] * self.n_train 
        self.ratings_per_user_test = [1] * self.n_test

        # Sparsity
        self.sparsity_train = 1 - len(train_values) / (self.n_train * len(self.ratings))
        self.sparsity_test = 1 - len(test_values) / (self.n_test * len(self.ratings))

        # Nombre d'utilisateurs actifs
        self.n_users_with_train = len(set(train_values))
        self.n_users_with_test = len(set(test_values))

        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'

    def _save_or_show(self, save_path=None):
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PDF saved: {save_path}")
        if self.show:
            plt.show()
        plt.close()

    def plot_rating_histograms(self, save_path=None):
        train_pct = [100 * self.train_counts.get(r, 0) / self.n_train for r in self.ratings]
        test_pct = [100 * self.test_counts.get(r, 0) / self.n_test for r in self.ratings]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        bars1 = ax1.bar(self.ratings, train_pct)
        ax1.set_title("Training Distribution")
        ax1.set_xlabel("Rating")
        ax1.set_ylabel("Percentage (%)")
        for b in bars1:
            if b.get_height() > 0.5:
                ax1.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.1f}%",
                         ha='center', va='bottom')

        bars2 = ax2.bar(self.ratings, test_pct)
        ax2.set_title("Test Distribution")
        ax2.set_xlabel("Rating")
        ax2.set_ylabel("Percentage (%)")
        for b in bars2:
            if b.get_height() > 0.5:
                ax2.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.1f}%",
                         ha='center', va='bottom')

        self._save_or_show(save_path)

    def plot_rating_comparison(self, save_path=None):
        train_pct = [100 * self.train_counts.get(r, 0) / self.n_train for r in self.ratings]
        test_pct = [100 * self.test_counts.get(r, 0) / self.n_test for r in self.ratings]

        x = np.arange(len(self.ratings))
        w = 0.35
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - w/2, train_pct, width=w, label="Train")
        ax.bar(x + w/2, test_pct, width=w, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.1f}" for r in self.ratings])
        ax.set_title("Rating Distribution Comparison")
        ax.legend()
        self._save_or_show(save_path)

    def run_all(self, save_dir=None):
        self.plot_rating_histograms(save_path=f"{save_dir}/rating_histograms.pdf" if save_dir else None)
        self.plot_rating_comparison(save_path=f"{save_dir}/rating_comparison.pdf" if save_dir else None)
        print("✓ Analysis complete!")






def plot_power_law_distribution(data_by_user, data_by_movie, save_path=None):
    def plot_power_law(degrees, label, color):
        values, counts = np.unique(degrees, return_counts=True)
        plt.plot(values, counts, marker='o', linestyle='', label=label, color=color)

    user_degrees = [len(ratings_list) for ratings_list in data_by_user]
    movie_degrees = [len(ratings_list) for ratings_list in data_by_movie]

    plt.figure(figsize=(8,6))
    plot_power_law(movie_degrees, label='Movies (items)', color='blue')
    plot_power_law(user_degrees, label='Users', color='green')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (number of ratings)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    if save_path is None:
        save_choice = input("Voulez-vous sauvegarder ce graphique en PDF ? (o/n) : ").strip().lower()
        if save_choice == 'o':
            save_path = input("Entrez le chemin complet du fichier PDF (ex: power_law.pdf) : ").strip()
    if save_path:
        plt.savefig(save_path, format='pdf')
        print(f"✓ Graphique sauvegardé : {save_path}")


## Function for rating distribnution

def plot_rating_distribution(data_by_user, save_path=None):
    plt.figure(figsize=(10, 6))
    
    all_ratings = [rating for ratings in data_by_user for _, rating in ratings]
    
    rating_counts = defaultdict(float)
    for r in all_ratings:
        rating_counts[r] += 1

    ratings = sorted(rating_counts.keys())
    counts = [rating_counts[r] for r in ratings]

    plt.bar(ratings, counts, width=0.45, color='skyblue', edgecolor='black')

    xmin, xmax = min(ratings), max(ratings)
    xticks = np.arange(xmin, xmax + 0.5, 0.5)
    plt.xticks(xticks)

    plt.xlabel('Rating')
    plt.ylabel('Number of Ratings')
    plt.title('Distribution of Ratings')

    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f" PDF saved: {save_path}")

    plt.show()




class RatingsDatasetAnalysis:
    """
    Analyse complète d'un dataset de ratings (train / test)
    """

    def __init__(
        self,
        data_by_user_train,
        data_by_user_test,
        data_by_movie_train,
        data_by_movie_test,
        sparsity_train,
        sparsity_test,
        show=True
    ):
        self.data_by_user_train = data_by_user_train
        self.data_by_user_test = data_by_user_test
        self.data_by_movie_train = data_by_movie_train
        self.data_by_movie_test = data_by_movie_test
        self.sparsity_train = sparsity_train
        self.sparsity_test = sparsity_test
        self.show = show


        self.train_ratings = np.array([
            rating for user in data_by_user_train for _, rating in user
        ])
        self.test_ratings = np.array([
            rating for user in data_by_user_test for _, rating in user
        ])

        self.n_train = len(self.train_ratings)
        self.n_test = len(self.test_ratings)
        self.n_total = self.n_train + self.n_test

  
        self.train_counts = Counter(self.train_ratings)
        self.test_counts = Counter(self.test_ratings)
        self.rating_values = sorted(set(self.train_ratings) | set(self.test_ratings))

        
        self.n_users_with_train = sum(len(u) > 0 for u in data_by_user_train)
        self.n_users_with_test = sum(len(u) > 0 for u in data_by_user_test)
        self.n_movies_with_train = sum(len(m) > 0 for m in data_by_movie_train)
        self.n_movies_with_test = sum(len(m) > 0 for m in data_by_movie_test)

        self.ratings_per_user_train = [len(u) for u in data_by_user_train]
        self.ratings_per_user_test = [len(u) for u in data_by_user_test]

        sns.set_style("whitegrid")
        plt.rcParams["figure.facecolor"] = "white"


    def _save_or_show(self, save_path=None):
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")
        if self.show:
            plt.show()
        plt.close()

    def plot_rating_histograms(self, save_path=None):
        train_pct = [100 * self.train_counts.get(r, 0) / self.n_train for r in self.rating_values]
        test_pct = [100 * self.test_counts.get(r, 0) / self.n_test for r in self.rating_values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.bar(self.rating_values, train_pct,width=0.3, edgecolor="black")
        ax1.set_title(f"Training Set (n={self.n_train:,})")
        ax1.set_xlabel("Rating")
        ax1.set_ylabel("Percentage (%)")

        ax2.bar(self.rating_values, test_pct, width=0.3, edgecolor="black", color="coral")
        ax2.set_title(f"Test Set (n={self.n_test:,})")
        ax2.set_xlabel("Rating")
        ax2.set_ylabel("Percentage (%)")

        self._save_or_show(save_path)


   
    def plot_rating_comparison(self, save_path=None):
        train_pct = [100 * self.train_counts.get(r, 0) / self.n_train for r in self.rating_values]
        test_pct = [100 * self.test_counts.get(r, 0) / self.n_test for r in self.rating_values]

        x = np.arange(len(self.rating_values))
        w = 0.35

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - w/2, train_pct, width=w, label="Train")
        ax.bar(x + w/2, test_pct, width=w, label="Test")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.1f}" for r in self.rating_values])
        ax.set_title("Rating Distribution: Train vs Test")
        ax.legend()

        self._save_or_show(save_path)

    def plot_ratings_per_user(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.hist(self.ratings_per_user_train, bins=30, edgecolor="black")
        ax1.set_title("Train: Ratings per User")
        ax1.axvline(np.mean(self.ratings_per_user_train), color="red", linestyle="--")

        ax2.hist(self.ratings_per_user_test, bins=30, edgecolor="black", color="coral")
        ax2.set_title("Test: Ratings per User")
        ax2.axvline(np.mean(self.ratings_per_user_test), color="red", linestyle="--")

        self._save_or_show(save_path)


    def plot_boxplot(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.boxplot(
            [self.train_ratings, self.test_ratings],
            labels=["Train", "Test"],
            patch_artist=True
        )
        ax.set_title("Rating Value Distribution")
        ax.set_ylabel("Rating")

        self._save_or_show(save_path)

    def plot_summary_dashboard(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Split
        axes[0, 0].pie(
            [self.n_train, self.n_test],
            labels=["Train", "Test"],
            autopct="%1.1f%%"
        )
        axes[0, 0].set_title("Train/Test Split")

        # Active users
        axes[0, 1].bar(
            ["Train", "Test"],
            [self.n_users_with_train, self.n_users_with_test]
        )
        axes[0, 1].set_title("Active Users")

        # Sparsity
        axes[1, 0].bar(
            ["Train", "Test"],
            [self.sparsity_train * 100, self.sparsity_test * 100]
        )
        axes[1, 0].set_title("Matrix Sparsity (%)")

     
        axes[1, 1].bar(
            ["Train", "Test"],
            [self.train_ratings.mean(), self.test_ratings.mean()]
        )
        axes[1, 1].set_title("Mean Rating")

        self._save_or_show(save_path)

    def summary(self):
      
        print(f"Train ratings: {self.n_train:,}")
        print(f"Test ratings : {self.n_test:,}")
        print(f"Mean rating (train): {self.train_ratings.mean():.3f}")
        print(f"Mean rating (test) : {self.test_ratings.mean():.3f}")
        print(f"Sparsity (train): {self.sparsity_train:.2%}")
        print(f"Sparsity (test) : {self.sparsity_test:.2%}")
      
    def run_all(self, save_dir=None):
        self.plot_rating_histograms(f"{save_dir}/rating_histograms.pdf" if save_dir else None)
        self.plot_rating_comparison(f"{save_dir}/rating_comparison.pdf" if save_dir else None)
        self.plot_ratings_per_user(f"{save_dir}/ratings_per_user.pdf" if save_dir else None)
        self.plot_boxplot(f"{save_dir}/rating_boxplot.pdf" if save_dir else None)
        self.plot_summary_dashboard(f"{save_dir}/summary_dashboard.pdf" if save_dir else None)
        self.summary()
        print(" Analysis complete!")
