from helpers import run_full_serial
from config_logmapres import logmap_rawdata_path
import config_logmapres as conf

if __name__ == "__main__":
    # 0. setting PCA parameters
    method = "random"
    method_params = conf.random_params
    method_name = method

    # run all lengths
    run_full_serial(logmap_rawdata_path, 
                    method, 
                    method_params, 
                    method_name, conf)     







