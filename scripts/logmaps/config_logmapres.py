""" configuration for results on ht logmap dataset"""
from scripts.config import project_path
import numpy as np

# data generation
logmapgen_params = dict(N=50,  # number of realizations
                        n=15_000,  # Length of time series
                        rint=(3.8, 4.),  # interval to chose from the value of r parameter
                        A0=np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [1, 0, 0]]),  # basic connection structure
                        A=np.array([[1., 0., 0.],
                                    [0.3, 1., 0.],
                                    [0.4, 0., 1.]]))


# Logmap output path
logmap_rawdata_path = project_path / 'data/logmap/logmap_rawdata.pkl'
interim_res_path = project_path  / 'results/interim/logmaps'
final_res_path = project_path / 'results/final'
figure_path = project_path / 'figures'

# common training parameters
train_split = 1. / 3
valid_split = 1. / 3
test_split = 1 - (train_split + valid_split)

# embedding params
embedding_params = dict(dimension=3, 
                        delay=1)

# method params
pca_params = dict(n_components=1)
ica_params = dict(n_components=2)
cca_params = dict(n_components=1)
dca_params = dict(d=1, T=5, n_init=10)
sfa_params = dict(n_components=1)
random_params = dict()
dcca_params = dict(input_size1=embedding_params['dimension'], 
                   input_size2=embedding_params['dimension'], 
                   n_components=1,
                   layer_sizes1=[20, 20, 1],
                   layer_sizes2=[20, 20, 1], 
                   epoch_num=500,
                   use_all_singular_values=True,
                   device='cpu')
shrec_params = dict(d_embed=embedding_params['dimension'],
                    n_components=1)
anisom_params = dict(
    init_params=dict(space_dim=embedding_params['dimension'], 
                     grid_dim=2, 
                     sizes=[40, 20]),
    fit_params = dict(epochs=1,
                      disable_tqdm=True)
                     )
                   

