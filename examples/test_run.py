import numpy as np
import matplotlib.pyplot as plt

import torch
from anisom.network.anisom import AniSOM
from anisom.datagen.logmap import gen_logmapdata
from anisom.preprocessing.tde import time_delay_embedding


# PARAMETERS
# Data generation parameters
logmapgen_params = dict(N=1,  # number of realizations
                        n=2_000,  # Length of time series
                        rint=(3.8, 4.),  # interval to chose from the value of r parameter
                        A0=np.array([[0, 0, 0],
                                     [1, 0, 0],
                                     [1, 0, 0]]),  # basic connection structure
                        A=np.array([[1., 0., 0.],
                                    [0.3, 1., 0.],
                                    [0.4, 0., 1.]]),
                        )


n_train = 100  # number of training samples
n_valid = 500  # number of validation samples
n_test = 500  # number of test samples

# embeddign parameters
d = 3
tau = 1
embedding_params = dict(
        dimension=d,
        delay=tau
        )

# Anisotrope SOM parameters
anisom_params = dict(
        space_dim=d,  # the input space dimension
        grid_dim=2,  # the number of dimensions of the grid
        sizes=[40, 20]  # grid size along each dimension
        )
fit_params = dict(
        epochs=2,
        disable_tqdm=False
        )

# SCRIPT
# Generate the data
dataset, params = gen_logmapdata(logmapgen_params)
x, y, z = dataset[0][:n_train, 1], dataset[0][:n_train, 2], dataset[0][:n_train, 0]
x_valid, y_valid, z_valid = dataset[0][n_train:n_train + n_valid, 1], dataset[0][n_train:n_train + n_valid, 2], dataset[0][n_train:n_train + n_valid, 0]
z_test, y_test, x_test = dataset[0][n_train + n_valid:, 0], dataset[0][n_train + n_valid:, 2], dataset[0][n_train + n_valid:, 1]

# Create time-delay embeddings
X = time_delay_embedding(x, **embedding_params)
Y = time_delay_embedding(y, **embedding_params)
X_valid = time_delay_embedding(x_valid, **embedding_params)
# Convert to torch tensors
X = torch.tensor(X)
Y = torch.tensor(Y)
X_valid = torch.tensor(X_valid)

# Create and fit the AniSOM model
model = AniSOM(**anisom_params)
model.fit(X, Y, **fit_params, x_valid=X_valid)