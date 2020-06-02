import numpy as np
import pandas as pd
import variables as vr


def retrieve_data():
    """
    retrieving task data
    :return:
    pandas data frame with data object
    """
    col_names = [f'hist_{i}' for i in range(27)] + ['img_name', 'img_path', 'target']
    return pd.read_csv(vr.DATA_PATH, compression='gzip', sep=' ', header=None, columns=col_names)

