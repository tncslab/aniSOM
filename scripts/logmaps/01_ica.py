from helpers import run_method, timeit

from sklearn.decomposition import FastICA
from src.savers.saver import save_results
from config_logmapres import logmapgen_params, logmap_rawdata_path, interim_res_path, embedding_params
import config_logmapres as conf
import pickle
from tqdm import tqdm
import os

if __name__ == "__main__":
    # some definitons and path creation for the results
    runner = timeit(run_method)
    os.makedirs(interim_res_path, exist_ok=True)

    # 0. setting PCA parameters
    method = FastICA
    method_params = conf.ica_params
    method_name = method.__name__
    N = logmapgen_params['N']
    
    # 1. Load raw data
    with open(logmap_rawdata_path, 'rb') as f:
        raw_dict = pickle.load(f)
    dataset, params = raw_dict['dataset'], raw_dict['params']

    # 2. Run PCA
    ms = []
    ts = []
    for i in tqdm(range(N)):
        data = dataset[i]
        m, t = runner(method=method,
                      data=data,
                      embedding_params=embedding_params,
                      method_params=method_params)
        ms.append(m)
        ts.append(t)

    # 3. save results
    df = save_results(fname=interim_res_path / f'{method_name}_res.csv',
                      r=ms,
                      times=ts,
                      N=N,
                      method=method_name,
                      dataset='logmap')







