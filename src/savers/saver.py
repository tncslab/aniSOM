import pandas as pd
import numpy as np


def save_results(fname, r, times, N, n, method, dataset):
    """Saves results to a file and returns the dataframe
    """
    df = pd.DataFrame(np.array([range(N),
                                r,
                                times,
                                N * [method],
                                N * [dataset],
                                N * [n]]).T,
                      columns=['data_id',
                               'r',
                               'duration',
                               'method',
                               'dataset',
                               'length'])
    df.to_csv(fname, index=False)
    return df
