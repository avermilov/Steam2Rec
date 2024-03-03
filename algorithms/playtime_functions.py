import warnings
import re
import math
import json
import glob
import random
from collections import Counter, defaultdict

import pathlib
import tqdm
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, csr_array, lil_array, lil_matrix, save_npz, load_npz


def flatten(xss):
    
    return [x for xs in xss for x in xs]


def train_test_split(interaction_matrix, test_size, index_to_appid, seed=42) -> (csr_matrix, dict):
    train_cols, train_rows, train_reviews = [], [], []
    test_interactions = dict()
    np.random.seed(seed)
    for i in tqdm.tqdm(range(interaction_matrix.shape[0])):
        game_idxs = interaction_matrix[i, :].indices
        reviews = interaction_matrix[i, :].data

        user_test_size = int(test_size * len(reviews))
        test_interactions[i] = [index_to_appid[idx] for idx in game_idxs[:user_test_size]]

        user_train_cols = game_idxs[user_test_size:]
        user_train_rows = [i] * len(user_train_cols)
        user_train_reviews = reviews[user_test_size:]

        train_cols.append(user_train_cols)
        train_rows.append(user_train_rows)
        train_reviews.append(user_train_reviews)
        
    return csr_matrix((flatten(train_reviews), (flatten(train_rows), flatten(train_cols))),
                                  shape=interaction_matrix.shape), test_interactions


def user_user_collaborative_filtering(interaction_matrix, user_index, k_neighbors=100, num_results=None):    
    # Calculate the user similarity matrix
    user_row = interaction_matrix[[user_index], :]
    user_similarity = cosine_similarity(user_row, interaction_matrix)
    user_similarity[0][user_index] = 0
    
    # Get the user's k nearest neighbors
    user_neighbors = np.argsort(user_similarity[0])[::-1][:k_neighbors]
    
    recommendations = user_neighbors
    if num_results is not None:
        recommendations = recommendations[:num_results]
        
    return recommendations


# this fast implementation requires precomputing similar games (items) for each game (see TODO)
def item_item_collaborative_filtering(similar_games_df, user_played_appids, num_results):
    user_played_appids = set(user_played_appids)
    similar_games_for_user = similar_games_df[similar_games_df.appid.isin(user_played_appids)]
    similar_unplayed_games = similar_games_for_user[~similar_games_for_user.similar_appid.isin(
        user_played_appids)].sort_values("cos_distance", ascending=False).drop_duplicates(
        subset=["similar_appid"], keep="first")
    
    return similar_unplayed_games.similar_appid.head(num_results).values


# Function for checking if values are included in lists and deleting rows
def check_values(row, developers, categories, genres):
    check1 = False
    check2 = False
    check3 = False
    
    if not pd.isnull(row['Developers']) and row['Developers'] in developers:
        check1 = True
    
    if not row['categories'] is np.nan:
        if any(value in categories for value in row['categories']):
            check2 = True
            
    if not row['genres'] is np.nan:
        if any(value in genres for value in row['genres']):
            check3 = True

    return check1 or check2 or check3


def get_popular_recommendations(game_info_df, interaction_matrix, user_index, INDEX_TO_APPID, num_results):
    game_info_df = game_info_df.copy()
    user_row = interaction_matrix[[user_index], :]
    
    appids_pop = [INDEX_TO_APPID[appid] for num, appid 
                  in enumerate(list(interaction_matrix[user_index].indices)) 
                  if user_row.data[num] != 0]
    developers, categories, genres = list(), list(), list()

    for appid in appids_pop:
        if not any(pd.isnull(game_info_df[game_info_df['AppID'] == appid]['Developers'].values)):
            developers += list(game_info_df[game_info_df['AppID'] == appid]['Developers'].values)
        if not any(pd.isnull(game_info_df[game_info_df['AppID'] == appid]['Categories'].values)):
            categories += list(game_info_df[game_info_df['AppID'] == appid]['Categories'].values)[0].split(',')
        if not any(pd.isnull(game_info_df[game_info_df['AppID'] == appid]['Genres'].values)):
            genres += list(game_info_df[game_info_df['AppID'] == appid]['Genres'].values)[0].split(',')
            
    developers = list(set(developers))
    categories = list(set(categories))
    genres = list(set(genres))
    
    
    filtered_df = game_info_df[game_info_df.apply(check_values, args=(developers, categories, genres), axis=1)]
    
    filtered_df["Rating"] = (filtered_df["Positive"] / (
        filtered_df["Positive"] + filtered_df["Negative"])).round(2)
    top_games = filtered_df.sort_values(["Rating", "TotalReviews"], ascending=False).head(num_results)
    recommendation = list(top_games["AppID"])
    
    return recommendation


def get_new_recommendations(game_info_df, interaction_matrix, user_index, INDEX_TO_APPID, num_results):
    game_info_df = game_info_df.copy()
    user_row = interaction_matrix[[user_index], :]
    
    appids_pop = [INDEX_TO_APPID[appid] for num, appid 
                  in enumerate(list(interaction_matrix[user_index].indices)) 
                  if user_row.data[num] != 0]
    developers, categories, genres = list(), list(), list()

    for appid in appids_pop:
        if not pd.isnull(game_info_df[game_info_df['AppID'] == appid]['Developers'].values):
            developers += list(game_info_df[game_info_df['AppID'] == appid]['Developers'].values)
        if not pd.isnull(game_info_df[game_info_df['AppID'] == appid]['Categories'].values):
            categories += list(game_info_df[game_info_df['AppID'] == appid]['Categories'].values)[0].split(',')
        if not pd.isnull(game_info_df[game_info_df['AppID'] == appid]['Genres'].values):
            genres += list(game_info_df[game_info_df['AppID'] == appid]['Genres'].values)[0].split(',')

    developers = list(set(developers))
    categories = list(set(categories))
    genres = list(set(genres))
    
    filtered_df = game_info_df[game_info_df.apply(check_values, args=(developers, categories, genres), axis=1)]
    
    filtered_df["Rating"] = (filtered_df["Positive"] / (
        filtered_df["Positive"] + filtered_df["Negative"])).round(2)
    filtered_df["Year"] = filtered_df['Release date'].astype(str).str[-4:]
    
    top_games = filtered_df.sort_values(["Year", "Rating", "TotalReviews"], ascending=False).head(num_results)
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
    
    return np.mean([average_precision_at_k(a, p, k) for a, p in zip(actuals, predicteds)])


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


def get_games_played(interaction_matrix, df, user_index, app_index_dict, verbose=False):
    rated_games = np.where(interaction_matrix[user_index].todense() != 0)[1]
    
    games_played = list()
    # see what games are recommended for the user
    for game_idx in rated_games:
        game_id = get_key(app_index_dict, game_idx)
        games_played.append(game_id)
        
        if verbose:
            print(df[df['AppID'] == game_id]['Name'].item())
        
    return games_played