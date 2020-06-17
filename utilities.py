import pandas as pd
import constants as cns
from scipy.stats import median_absolute_deviation as mad
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def retrieve_data():
    """
    retrieving and basic preparation of data task data
    Delete columns with names and paths to images as useless mostly
    :return: pandas data frame with data object
    """
    col_names = [f'hist_{i}' for i in range(27)] + ['img_name', 'img_path', 'target']
    collected_data = pd.read_csv(cns.DATA_PATH, compression='gzip', sep=' ', header=None, names=col_names)
    collected_data.drop(columns=['img_name', 'img_path'], inplace=True)
    map_dic = {'Inlier': 0, 'Outlier': 1}
    collected_data.target.replace(map_dic, inplace=True)
    return collected_data


def extract_features(data_set):
    """
    extracting the set of features from data set
    :param data_set: pandas data frame
    :return: pandas data frame
    """
    return data_set[[f'hist_{i}' for i in range(27)]]


def describe_data(data_set):
    """
    basic descriptive statistic calculation
    descr method with added median_absolute_deviation
    :param data_set: pandas data frame
    :return:
    pandas data frame with descriptive stats
    """
    describe_stats = data_set.describe().reset_index()
    describe_stats.loc[8] = ['mad'] + [mad(data_set[f'hist_{i}']) for i in range(27)] + [0]
    describe_stats.set_index('index', inplace=True, drop=True)
    assert isinstance(describe_stats, object)
    return describe_stats


def calculate_tsne(data_set):
    """
    calculate 2 dimension tsne transformation with pca initialisation
    :param data_set: pandas data frame
    :return: numpy array
    """
    transformer = TSNE(n_components=2, init='pca', n_jobs=-1)
    return transformer.fit_transform(data_set)


def calculate_pca(data_set):
    """
    calculate 2 dimension pca transformation
    :param data_set: pandas data frame
    :return: numpy array
    """
    transformer = PCA(n_components=2)
    return transformer.fit_transform(data_set)


def plot_low_dimensions(low_dim_data, labels):
    """
    plot the 2 dimensional representation of data set
    :param low_dim_data: array like 2 dimensional data
    :param labels: array like labels for data
    :return:
    """
    low_dim_df = pd.DataFrame(low_dim_data, columns=['x', 'y'])
    plt.figure(figsize=(20, 20))
    return sns.scatterplot(x='x', y='y', data=low_dim_df, hue=labels, palette='Set2')


def plot_results(roc_curve, pr_curve):
    """
    plot the supervised learning metrics of ROC curve and PR curve
    :param roc_curve: list with 2 dimesions of data points
    :param pr_curve: list with 2 dimensions of data points
    :return:
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].set_title("ROC Curve")
    ax[0].plot([0, 1], [0, 1], linestyle="--", label="No Skill")
    ax[0].plot(roc_curve[0], roc_curve[1], marker='.', label="ROC curve")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].legend()
    ax[1].set_title("PR Curve")
    ax[1].plot([0, 1], [1, 0], linestyle="--", label="No Skill")
    ax[1].plot(pr_curve[1], pr_curve[0], marker=".", label="PR curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend()


if __name__ == '__main__':
    try:
        data = retrieve_data()
        if data.shape == (50000, 28):
            print("Seems that's all works")
        else:
            print("Something went wrong with data")
    except FileNotFoundError:
        print('File seems to be unreachable at the moment')
