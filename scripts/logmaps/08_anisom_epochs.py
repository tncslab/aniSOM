import sys
sys.path.append('./')
sys.path.append('../../')
from src.network.anisom import AniSOM
from helpers import run_full_parallel
from config_logmapres import logmap_rawdata_path
import config_logmapres as conf


if __name__ == "__main__":
    # 0. setting PCA parameters
    method = AniSOM
    method_params = conf.anisom_multiepoch_params['init_params']
    method_fit_params = conf.anisom_multiepoch_params['fit_params']
    method_name = method.__name__

    # run all lengths
    run_full_parallel(logmap_rawdata_path, 
                      method, 
                      method_params, 
                      method_name,
                      conf,
                      fit_params=method_fit_params)  
