"""This script generates the logisticmap dataset
"""

import sys
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../../../')
from src.datagen.logmap import gen_logmapdata
from config_logmapres import logmapgen_params, logmap_rawdata_path
import pickle

if __name__ == "__main__":
    # 1. Generate data
    N = logmapgen_params['N']  # number of realizations
    dataset, params = gen_logmapdata(logmapgen_params)

    # 2. Save the data
    with open(logmap_rawdata_path, 'wb') as f:
        pickle.dump({'dataset': dataset, 'params': params}, f)
    print(f"Generated {N} logmap realizations and saved to {logmap_rawdata_path}")

