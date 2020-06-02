import numpy as np
import pandas as pd
import variables as vr


def retrieve_data():
    """
    retrieving and basic preparation of data task data
    :return:
    pandas data frame with data object
    """
    col_names = [f'hist_{i}' for i in range(27)] + ['img_name', 'img_path', 'target']
    data = pd.read_csv(vr.DATA_PATH, compression='gzip', sep=' ', header=None, names=col_names)
    data = data.drop(columns=['img_name', 'img_path'])
    map_dic = {'Inlier': 0, 'Outlier': 1}
    data.target.replace(map_dic, inplace=True)
    return data

