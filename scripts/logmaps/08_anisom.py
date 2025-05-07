from helpers import run_method

from src.network.anisom import AniSOM
from src.savers.saver import save_results
from config_logmapres import logmapgen_params, logmap_rawdata_path, interim_res_path, embedding_params
import config_logmapres as conf
import pickle
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial


if __name__ == "__main__":
    # some definitons and path creation for the results
    os.makedirs(interim_res_path, exist_ok=True)

    # 0. setting PCA parameters
    method = AniSOM
    method_params = conf.anisom_params['init_params']
    method_fit_params = conf.anisom_params['fit_params']
    method_name = method.__name__
    N = logmapgen_params['N']
    # print(method_name)
    # exit()
    
    # 1. Load raw data
    with open(logmap_rawdata_path, 'rb') as f:
        raw_dict = pickle.load(f)
    dataset, params = raw_dict['dataset'], raw_dict['params']

    # 2. Run Parallel
    runner = partial(run_method,
                     method=method,
                     embedding_params=embedding_params,
                     method_params=method_params,
                     fit_params=method_fit_params)

    with Pool(processes=os.cpu_count()) as pool:
        tm = pool.map(runner, dataset)
    ms, ts = zip(*tm)

    # 3. save results
    df = save_results(fname=interim_res_path / f'{method_name}_res.csv',
                      r=ms,
                      times=ts,
                      N=N,
                      method=method_name,
                      dataset='logmap')