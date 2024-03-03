from collections import Counter

import numpy as np
import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def flatten(xss):
    return [x for xs in xss for x in xs]


def train_test_split(
    interaction_matrix, test_size, index_to_appid, seed=42
) -> (csr_matrix, dict):
    train_cols, train_rows, train_reviews = [], [], []
    test_interactions = dict()
    np.random.seed(seed)
    for i in tqdm.tqdm(range(interaction_matrix.shape[0])):
        game_idxs = interaction_matrix[i, :].indices
        reviews = interaction_matrix[i, :].data
        pos_reviews_idxs, neg_reviews_idxs = list(game_idxs[reviews == 1]), list(
            game_idxs[reviews == -1]
        )

        user_test_size = int(test_size * len(reviews))
        if len(pos_reviews_idxs) < user_test_size:
            train_cols.append(game_idxs)
            train_rows.append([i] * len(game_idxs))
            train_reviews.append(reviews)
            continue

        np.random.shuffle(pos_reviews_idxs)

        test_interactions[i] = [
            index_to_appid[idx] for idx in pos_reviews_idxs[:user_test_size]
        ]

        user_train_cols = pos_reviews_idxs[user_test_size:] + neg_reviews_idxs
        user_train_rows = [i] * len(user_train_cols)
        user_train_reviews = [1] * (len(pos_reviews_idxs) - user_test_size) + [
            -1
        ] * len(neg_reviews_idxs)

        train_cols.append(user_train_cols)
        train_rows.append(user_train_rows)
        train_reviews.append(user_train_reviews)

    return (
        csr_matrix(
            (flatten(train_reviews), (flatten(train_rows), flatten(train_cols))),
            shape=interaction_matrix.shape,
        ),
        test_interactions,
    )


def user_user_collaborative_filtering(
    interaction_matrix, user_index, k_neighbors=100, num_results=None
):
    # Calculate the user similarity matrix
    user_row = interaction_matrix[[user_index], :]
    user_similarity = cosine_similarity(user_row, interaction_matrix)
    user_similarity[0][user_index] = 0

    # Get the user's k nearest neighbors
    user_neighbors = np.argsort(user_similarity[0])[::-1][:k_neighbors]

    # Calculating like ratios and number of reviews for each game among neighbors
    user_ratings = []
    for i in range(interaction_matrix[user_neighbors].shape[1]):
        game_ratings = interaction_matrix[user_neighbors][:, i]

        # If at least one neighbor rated the game, calculate liked ratio
        # else, assign it 0
        neighbor_average_rating = 0
        num_neighbors_reviewed = game_ratings.data.shape[0]
        if num_neighbors_reviewed != 0:
            c = Counter(game_ratings.data)
            pos = c[1] if 1 in c else 0
            neg = c[-1] if -1 in c else 0
            neighbor_average_rating = pos / (pos + neg)
        user_ratings.append((num_neighbors_reviewed, round(neighbor_average_rating, 2)))
    user_ratings = np.array(user_ratings)

    selected_games = np.where(interaction_matrix[user_index].todense() == 0)[1]
    user_ratings = list(map(tuple, user_ratings[selected_games]))
    # Sort by most reviews, then by like ratio
    recommendations = sorted(
        zip(selected_games, user_ratings), key=lambda x: x[1], reverse=True
    )
    if num_results is not None:
        recommendations = recommendations[:num_results]

    recommendations = [i[0] for i in recommendations]

    return recommendations


def item_item_collaborative_filtering(
    similar_games_df, user_played_appids, num_results
):
    user_played_appids = set(user_played_appids)
    similar_games_for_user = similar_games_df[
        similar_games_df.appid.isin(user_played_appids)
    ]
    similar_unplayed_games = (
        similar_games_for_user[
            ~similar_games_for_user.similar_appid.isin(user_played_appids)
        ]
        .sort_values("cos_distance", ascending=False)
        .drop_duplicates(subset=["similar_appid"], keep="first")
    )

    return similar_unplayed_games.similar_appid.head(num_results).values


def get_popular_recommendations(game_info_df, num_results):
    game_info_df = game_info_df.copy()
    game_info_df["Rating"] = (
        game_info_df["Positive"] / (game_info_df["Positive"] + game_info_df["Negative"])
    ).round(2)
    top_games = game_info_df.sort_values(
        ["Rating", "TotalReviews"], ascending=False
    ).head(num_results)
    recommendations = list(top_games["AppID"])

    return recommendations


def get_new_recommendations(game_info_df, num_results):
    game_info_df = game_info_df.copy()
    game_info_df["Rating"] = (
        game_info_df["Positive"] / (game_info_df["Positive"] + game_info_df["Negative"])
    ).round(2)
    game_info_df["Year"] = game_info_df["Release date"].astype(str).str[-4:]
    top_games = game_info_df.sort_values(
        ["Year", "Rating", "TotalReviews"], ascending=False
    ).head(num_results)
    recommendation = list(top_games["AppID"])

    return recommendation


def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    num_actual = len(actual_set)
    if num_actual < k:
        k = num_actual

    num_found = len(actual_set.intersection(set(predicted[:k])))
    precision = num_found / k

    return precision


def average_precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    num_actual = len(actual_set)
    if num_actual < k:
        k = num_actual

    cum_sum = ap = 0
    for i in range(k):
        if predicted[i] in actual_set:
            cum_sum += 1
            ap += cum_sum / (i + 1)

    return ap / k


def mean_average_precision_at_k(actuals, predicteds, k):
    return np.mean(
        [average_precision_at_k(a, p, k) for a, p in zip(actuals, predicteds)]
    )


def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    num_actual = len(actual_set)
    if num_actual < k:
        k = num_actual

    num_found = len(actual_set.intersection(set(predicted[:k])))
    recall = num_found / num_actual

    return recall


def ndcg_at_k(actual, predicted, k):
    actual_set = set(actual)
    k = min(k, len(actual))
    # Calculating IDCG (Ideal Discounted Cumulative Gain)
    idcg = (1 / (np.log2(np.arange(2, k + 2)))).sum()

    # Calculating DCG (Discounted Cumulative Gain)
    if len(predicted) < k:
        predicted = predicted + [-1] * (k - len(predicted))
    dcg = np.sum([1 / np.log2(i + 2) for i in range(k) if predicted[i] in actual_set])

    # Calculating nDCG
    return dcg / idcg if idcg != 0 else 0


def mean_ndcg_at_k(actuals, predicteds, k):
    return np.mean([ndcg_at_k(a, p, k) for a, p in zip(actuals, predicteds)])
