import time
import numpy as np
import sys
sys.path.append('./')
sys.path.append('../../')
sys.path.append('../../../')
from src.preprocessing.splitters import train_valid_test_split
from src.preprocessing.tde import time_delay_embedding

from src.evaluate.evalz import comp_ccorr, get_maxes
from config_logmapres import train_split, interim_res_path, valid_split

# optional imports that are needed only for specific methods
try:
    import torch
except:
     print("pytorch not installed")

try:
    from sklearn.preprocessing import PolynomialFeatures
except:
    print("sklearn not installed")




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
            if method.__name__ in ['PCA', 'FastICA', 'DynamicalComponentsAnalysis', 'SFA']: 
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
                print(X_train.shape, Y_train.shape)
                # exit()
                model.fit(X_train, Y_train)
                zpred, zpred2 = model.transform(X_test, Y_test)

            elif method.__name__ == 'DCCA':
                model.fit([X_train, Y_train])
                zpred, zpred2 = model.transform([X_test, Y_test])

            

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
