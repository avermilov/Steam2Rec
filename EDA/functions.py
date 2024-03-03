import glob
import math
import re
import time
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

# functions of EDA_libraries


def describe_perc(arr):
    return pd.Series(arr).describe(
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )


def plot(data, ylabel, xlabel, title):
    plt.figure(figsize=(15, 10))
    plt.hist(data, bins=200)
    plt.grid()
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.title(title, fontsize=20)
    plt.show()


def filter_data(data, lower_q, upper_q):
    left_border = (np.quantile(data, lower_q),)
    right_border = np.quantile(data, upper_q)

    return data[np.where((data >= left_border) & (data <= right_border))]


def group_data(data, feature, group_function):
    if group_function == "size":
        return data.groupby(feature).size().values

    elif group_function == "mean":
        return data.groupby(feature)["playtime_minutes"].mean().values


# functions of EDA_games


def get_set_of_smth(smth):
    if "nan" in str(smth).lower():
        return np.nan

    else:
        return set(str(smth).split(","))


def make_histplot(feature, data, x_label, rotation=False):
    sns.histplot(data[feature])
    if rotation:
        plt.xticks(rotation=90)
    plt.title(f"Distribution of the feature {feature}")
    plt.ylabel("Number of games")
    plt.xlabel(x_label)
    plt.show()


def make_histplot_filtr(feature, data_list, x_labels, rotation=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    filtr = [
        "no restrictions",
        "over 50 reviews",
        "over 70 reviews",
        "over 100 reviews",
    ]
    for num, data in enumerate(data_list):
        ax = axs[num // 2, num % 2]
        sns.histplot(data[feature], ax=ax)
        if rotation:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_title(f"{filtr[num]}")

        if num < 2:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xlabel(x_labels)

        if num % 2 > 0:
            ax.set_ylabel("")

        else:
            ax.set_ylabel("Number of games")

    fig.suptitle(f"Distribution of the feature {feature} \n", fontsize=16)

    plt.tight_layout()
    plt.show()


def make_countplot(feature, data, x_label, order, rotation=False):
    sns.countplot(x=feature, data=data, order=order)
    if rotation:
        plt.xticks(rotation=90)
    plt.title(f"Distribution of the feature {feature}")
    plt.ylabel("Number of games")
    plt.xlabel(x_label)
    plt.show()


def make_countplot_filtr(
    feature, data_list, x_labels, order, rotation=False, limit=False
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    filtr = [
        "no restrictions",
        "over 50 reviews",
        "over 70 reviews",
        "over 100 reviews",
    ]
    for num, data in enumerate(data_list):
        ax = axs[num // 2, num % 2]
        sns.countplot(x=feature, data=data, order=order, ax=ax)

        if rotation:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title(f"{filtr[num]}")

        if num < 2:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xlabel(x_labels)

        if num % 2 > 0:
            ax.set_ylabel("")

        else:
            ax.set_ylabel("Number of games")

        if limit:
            ax.set_xlim(-1, limit + 0.5)

    fig.suptitle(f"Distribution of the feature {feature} \n", fontsize=16)

    plt.tight_layout()
    plt.show()


def sort_range(ranges):
    def extract_range(estimated_owners):
        if estimated_owners == "0 - 0":
            return 0
        else:
            start, end = map(int, estimated_owners.split(" - "))
            return (start + end) / 2

    return sorted(ranges.unique(), key=extract_range)


def sort_prices(prices):
    def extract_price(price_group):
        if price_group == "Free":
            return 0
        if price_group.startswith("<"):
            return int(price_group.lstrip("< ").rstrip("$"))
        if price_group.startswith(">"):
            return int(price_group.lstrip("> ").rstrip("$")) + 1
        else:
            return int(price_group.rstrip("$"))

    return sorted(prices.unique(), key=extract_price)


def make_distplot(
    feature, data, x_label, rotation=False, x_min=-math.inf, x_max=math.inf
):
    sns.distplot(data[feature][(data[feature] > x_min) & (data[feature] < x_max)])
    if rotation:
        plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.title(f"Distribution of the feature {feature}")
    plt.show()


def make_distplot_filtr(
    feature, data_list, x_labels, rotation=False, x_min=-math.inf, x_max=math.inf
):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    filtr = [
        "no restrictions",
        "over 50 reviews",
        "over 70 reviews",
        "over 100 reviews",
    ]
    for num, data in enumerate(data_list):
        ax = axs[num // 2, num % 2]
        sns.distplot(
            data[feature][(data[feature] > x_min) & (data[feature] < x_max)], ax=ax
        )

        if rotation:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title(f"{filtr[num]}")

        if num < 2:
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xlabel(x_labels)

        if num % 2 > 0:
            ax.set_ylabel("")

        else:
            ax.set_ylabel("Number of games")

    fig.suptitle(f"Distribution of the feature {feature} \n", fontsize=16)

    plt.tight_layout()
    plt.show()


def make_barplot(feature, data, x_label, title, rotation=False, pre_counted=False):
    if pre_counted:
        plt.bar(data[feature], data["Count"])
    else:
        data = data[feature].explode().value_counts().reset_index()
        data = data[data[feature] != ""]
        plt.bar(data.iloc[:10, 0], data.iloc[:10, 1])

    if rotation:
        plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel("Number of games")
    plt.title(f"Number of games by {title}")
    plt.show()


def make_barplot_filtr(feature, data_list, x_label, title, rotation=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()
    filtr = [
        "no restrictions",
        "over 50 reviews",
        "over 70 reviews",
        "over 100 reviews",
    ]

    for num, data in enumerate(data_list):
        data = data[feature].explode().value_counts().reset_index()
        axs[num].bar(data.iloc[:10, 0], data.iloc[:10, 1])
        if rotation:
            axs[num].set_xticklabels(data.iloc[:10, 0], rotation=90)

        axs[num].set_title(f"{filtr[num]}")
        axs[num].set_xlabel(x_label)

        if num % 2 > 0:
            axs[num].set_ylabel("")

        else:
            axs[num].set_ylabel("Number of games")

    fig.suptitle(f"Number of games by {title} \n", fontsize=16)

    plt.tight_layout()
    plt.show()


def countplot_many_features(data, feature, hue, order, rotation=False):
    data = data[data[feature].isin(order)]
    sns.countplot(data=data, x=feature, hue=hue, order=order)
    if rotation:
        plt.xticks(rotation=90)
    plt.xlabel(feature)
    plt.ylabel("Number of games")
    plt.title(f"Number of games by {feature}")
    plt.show()


# Исправление формата даты в столбце Release_date


def date_change_format(date):
    month_names = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    month_nums = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]
    new_date = ""
    for month_name, month_num in zip(month_names, month_nums):
        if month_name in str(date).lower():
            new_date += month_num
            break
    new_date += " "
    new_date += date[-4:]

    return datetime.strptime(new_date, "%m %Y")


def make_price_groups(price):
    for price_group in range(0, 80, 5):
        if float(price) == 0:
            return "Free"
        if price <= price_group:
            return "< " + str(price_group) + "$"

    return "> 80"


def make_set(feature, new_feature, df):
    df[new_feature] = [get_set_of_smth(i) for i in df[feature]]

    all_text = list()
    for i in df[feature]:
        if "nan" not in str(i).lower():
            new_text_list = re.sub("[^a-z0-9,]", "", str(i).lower()).split(",")
            all_text.extend(new_text_list)

    dict_text = Counter(all_text)

    sorted_text = sorted(dict_text.items(), key=lambda item: item[1], reverse=True)
    sorted_text_df = pd.DataFrame(sorted_text[0:10], columns=[feature, "Count"])

    return df, sorted_text_df


def get_notes_type(note):
    sex_words = [
        "sex",
        "erotic",
        "nake",
        "nud",
        "underwear",
        "butt",
        "cloth",
        "fetish",
        "girl",
        "kiss",
        "masturb",
        "femal",
    ]
    violence_words = [
        "violen",
        "blood",
        "shoot",
        "kill",
        "abus",
        "gore",
        "weapons",
        "destr",
        "monster",
        "fight",
    ]
    mature_words = ["matur", "mild", "shock", "all ages", "child", "nsfw", "languag"]
    depression_words = ["suicid", "depression", "addiction", "smok", "alcoh"]

    sex_pattern = "(?:{})".format("|".join(sex_words))
    violence_pattern = "(?:{})".format("|".join(violence_words))
    mature_pattern = "(?:{})".format("|".join(mature_words))
    depression_pattern = "(?:{})".format("|".join(depression_words))
    none_pattern = "(?:{})".format("|".join(["none", "nan"]))

    patterns = [
        none_pattern,
        sex_pattern,
        violence_pattern,
        depression_pattern,
        mature_pattern,
    ]
    game_types = [
        "all ages",
        "sexual content",
        "violent content",
        "depression/suicide content",
        "mature content",
    ]

    new_note = str(note).lower()

    for pattern, game_type in zip(patterns, game_types):
        if bool(re.search(pattern, new_note)):
            return game_type


def plot_correlated_features(df, feature1, feature2):
    plt.scatter(df[feature1], df[feature2])
    plt.title(f"Correlated features \n{feature1} и {feature2}")
    plt.show()


# EDA reviews


class FeatureCounter:
    def __init__(
        self,
        feature,
        n_jobs,
        condition=False,
        game_review_files=glob.glob("SteamReviewsCombined/*.csv"),
    ):
        self.feature = feature
        self.n_jobs = n_jobs
        self.condition = condition
        self.game_review_files = game_review_files

    # Для группировки по играм
    # Создаем функцию, которая будет выполняться отдельно для каждого файла-игры

    def get_file_counts(self, file_path) -> Counter:
        df = pd.read_csv(file_path)
        counts = Counter(df[self.feature])
        return counts

    def get_feature_counts(self):
        start_time = time.time()
        if self.condition:
            games_feature_counts = Parallel(n_jobs=self.n_jobs)(
                delayed(self.get_file_counts)(game_file_path)
                for game_file_path in self.game_review_files
                if int(re.sub("[^0-9]", "", str(game_file_path).lower()))
                in df_popular["AppID"]
            )
        else:
            games_feature_counts = Parallel(n_jobs=self.n_jobs)(
                delayed(self.get_file_counts)(game_file_path)
                for game_file_path in self.game_review_files
            )
        end_time = time.time()
        print("Total time:", end_time - start_time)
        return games_feature_counts

    def get_all_feature_counts(self):
        # Скомбинируем все счетчики в один общий счетчик
        games_feature_counts = self.get_feature_counts()
        total_counter = games_feature_counts[0]
        for counter in games_feature_counts[1:]:
            total_counter += counter
        return total_counter


def get_avg_by_feature(df, feature, n_jobs, bool_feature=True):
    _counter = FeatureCounter(feature, n_jobs)
    _list = _counter.get_feature_counts()

    game_feature = list()
    if bool_feature:
        for i in _list:
            game_feature.append(i[1] / (i[0] + i[1]))
    else:
        for i in range(len(_list)):
            sum_els = 0
            sum_weighted = 0
            for i, j in zip(_list[i], _list[i].values()):
                sum_weighted += i * j
                sum_els += j
            game_feature.append(sum_weighted / sum_els)

    df[feature] = game_feature

    return df


def get_data(feature_list, file_path: str):
    df = pd.read_csv(file_path)

    return df[["author_steamid"] + feature_list]
