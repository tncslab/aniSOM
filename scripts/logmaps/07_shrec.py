from shrec.models import RecurrenceManifold
from helpers import run_full_serial
from config_logmapres import logmap_rawdata_path
import config_logmapres as conf

if __name__ == "__main__":
    # 0. setting PCA parameters
    method = RecurrenceManifold
    method_params = conf.shrec_params
    method_name = method.__name__

    # run all lengths
    run_full_serial(logmap_rawdata_path, 
                    method, 
                    method_params, 
                    method_name, conf)   
    