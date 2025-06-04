import time
import numpy as np
import sys
sys.path.append('./')
sys.path.append('../../')
sys.path.append('../../../')
from src.preprocessing.splitters import train_valid_test_split
from src.preprocessing.tde import time_delay_embedding
from src.savers import save_results

from src.evaluate.evalz import comp_ccorr, get_maxes
from config_logmapres import train_split, interim_res_path, valid_split

from tqdm import tqdm

import pickle
import os

from functools import partial
from multiprocessing import Pool

# optional imports that are needed only for specific methods
try:
    import torch
except:
     print("pytorch not installed")

try:
    from sklearn.preprocessing import PolynomialFeatures
except:
    print("sklearn not installed")


def normalize(x):
    return (x - x.mean()) / x.std()
    

def aggregate_preds(zpred1 : np.ndarray, zpred2 : np.ndarray) -> np.ndarray:
    """Aggregates two estimates of z by taking the mean.
    But it can happen that the two estimates are anticorrelated, so the has to be normalized and oriented into the same direction.


    """
    # normalize time series
    z1 = normalize(zpred1)
    z2 = normalize(zpred2)

    # check if orientation matches
    if np.dot(z1.T, z2) < 0:
        z2 = -z2

    return (z1 + z2) / 2



def extract_vars(data : np.ndarray, embedding_params: dict) -> tuple:
    """Extracts the variables from the data.

    Args:
        data (np.ndarray): The data.
        embedding_params (dict): The embedding parameters.

    Returns:
        tuple: The extracted variables
    
    """
    d_embed = embedding_params['dimension']

    X = time_delay_embedding(data[:, 1], **embedding_params)
    Y = time_delay_embedding(data[:, 2], **embedding_params)
    z = data[d_embed - 1:, 0]
    return X, Y, z


def run_method(data, method, embedding_params, method_params, fit_params=None):
        start = time.time()
        X, Y, z = extract_vars(data, embedding_params)

        X_train, Y_train, z_train, X_valid, Y_valid, z_valid, X_test, Y_test, z_test = train_valid_test_split(X, Y, z, 
                                                                                                              train_split, 
                                                                                                              valid_split)

        if method == 'random':
             n_components = 1
             zpred = np.random.rand(len(z_test)).reshape(-1, 1)

        else:
            if method.__name__=="DCCA": 
                def myfun(x, *args, **kwargs):
                    return torch.linalg.eigh(x)

                torch.symeig = myfun  # redefine function to make it work with dcca

            # model definition
            model = method(**method_params)

            # if the model is PCA, ICA, DCA 
            try:
                n_components = method_params['n_components']
            except:
                n_components = 1
            if method.__name__ in ['PCA', 'FastICA', 'DynamicalComponentsAnalysis', 'SFA', "KernelPCA"]: 
                D = np.concatenate([X_train, Y_train], axis=1)
                D_test = np.concatenate([X_test, Y_test], axis=1)

                if method.__name__ == 'SFA':
                    # Create Polynomial features
                    poly = PolynomialFeatures(degree=2)
                    D = poly.fit_transform(D)
                    D_test = poly.transform(D_test)

                model = model.fit(D)
                zpred = model.transform(D_test)
            
            elif method.__name__ in ['CCA']:
                model.fit(X_train, Y_train)
                zpred1, zpred2 = model.transform(X_test, Y_test)
                zpred = aggregate_preds(zpred1, zpred2)


            elif method.__name__ == 'DCCA':
                model.fit([X_train, Y_train])
                zpred1, zpred2 = model.transform([X_test, Y_test])
                zpred = aggregate_preds(zpred1, zpred2)

            

            elif method.__name__ == 'RecurrenceManifold':
                X = data[:, 1:]
                z = data[:, 0]

                X_train, _, z_train, X_valid, _, z_valid, X_test, __, z_test = train_valid_test_split(X, X, z,
                                                                                                    train_split,
                                                                                                    valid_split)
                zpred = model.fit_predict(X_train).reshape(-1, 1)
                z_test = z_train  # set the set to z_train, because cannot predict without fit


            elif method.__name__ == 'AniSOM':
                n_components = 1
                model.fit(torch.Tensor(X_train), torch.Tensor(Y_train), **fit_params)
                zpred = model.predict(torch.Tensor(X_test))[:, 1:]
            
            else:
                raise NotImplementedError

        maxc = max([get_maxes(*comp_ccorr(z_test, zpred[:, j]))[1] for j in range(n_components)])
        end = time.time()
        return maxc, end - start

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        return result
    return wrapper


def run_full_serial(rawdata_path, method, method_params, method_name, conf, fit_params=None):


    # ectract parameters
    N = conf.logmapgen_params['N']
    embedding_params = conf.embedding_params
    interim_res_path = conf.interim_res_path

    # create folder if not exist
    os.makedirs(interim_res_path, exist_ok=True)

    # 1. Load raw data
    with open(rawdata_path, 'rb') as f:
        raw_dict = pickle.load(f)
    raw_dataset, params = raw_dict['dataset'], raw_dict['params']


    for n in tqdm(conf.ns):
        dataset = [data[:n] for data in raw_dataset]

        # 2. Run PCA
        ms = []
        ts = []
        for i in tqdm(range(N)):
            data = dataset[i]
            m, t = run_method(
                method=method,
                data=data,
                embedding_params=embedding_params,
                method_params=method_params,
                fit_params=fit_params)
            ms.append(m)
            ts.append(t)

        # 3. save results
        df = save_results(fname=interim_res_path / f'{method_name}_{n}_res.csv',
                          r=ms,
                          times=ts,
                          N=N,
                          n=n,
                          method=method_name,
                          dataset='logmap')

def run_full_parallel(rawdata_path, method, method_params, method_name, conf, fit_params=None):
    # ectract parameters
    N = conf.logmapgen_params['N']
    embedding_params = conf.embedding_params
    interim_res_path = conf.interim_res_path

    # create folder if not exist
    os.makedirs(interim_res_path, exist_ok=True)

    # 1. Load raw data
    with open(rawdata_path, 'rb') as f:
        raw_dict = pickle.load(f)
    raw_dataset, params = raw_dict['dataset'], raw_dict['params']


    for n in tqdm(conf.ns):
        dataset = [data[:n] for data in raw_dataset]

        # put in a conditional if the method is AniSOM class then set the number of epochs to be approximately the same
        if method.__name__ == 'AniSOM':
            fit_params['epochs'] = np.ceil(20_000 / (n * train_split) ).astype(int)
            print("Set epochs to", fit_params['epochs'])


        runner = partial(run_method,
                         method=method,
                         embedding_params=embedding_params,
                         method_params=method_params,
                         fit_params=fit_params)
        
        with Pool(processes=os.cpu_count()) as pool:
            tm = pool.map(runner, dataset)
        ms, ts = zip(*tm)

        # 3. save results
        df = save_results(fname=interim_res_path / f'{method_name}_{n}_res.csv',
                          r=ms,
                          times=ts,
                          N=N,
                          n=n,
                          method=method_name,
                          dataset='logmap')