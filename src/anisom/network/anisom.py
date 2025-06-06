import torch
import torch.nn as nn
import numpy as np
from scipy.spatial  import cKDTree
from tqdm.auto import tqdm

from anisom.preprocessing.tde import TimeDelayEmbeddingTransform
import matplotlib.pyplot as plt


class AniSOM(nn.Module):
    """
    Anisotropic Self-Organizing Map (AniSOM).

    This class implements an Anisotropic Self-Organizing Map, a type of neural network
    that is trained using unsupervised learning to reconstruct the discretized state of 
    shared driver dynamics from parallel samples of the affected systems.
    The neighborhoods found in the Y variable are guiding the training of the SOM on the X variable, 
    such that the own-dynamics and the hidden driver dynamics is disentangled on the resulting 2D grid.

    """
    def __init__(self, space_dim : int, grid_dim : int, sizes : list[int]):
        """Anisotropic Self-Organizing Map

        :param space_dim: embedding space dimension
        :param grid_dim: 2D is supported
        :param sizes: list of sizes for the 2 grid dimensions
        """
        super(AniSOM, self).__init__()
        self.grid_dim = grid_dim
        self.sizes = sizes
        self.grid_shape = sizes + [space_dim]
        # self.grid = 0.5 + 0.3 * torch.randn(np.prod(self.grid_shape)).reshape(self.grid_shape)
        i_vals = 1/self.sizes[0] * torch.arange(self.sizes[0]).view([-1, 1]).expand(-1, self.sizes[1])
        j_vals = 1/self.sizes[1] * torch.arange(self.sizes[1]).view([1, -1]).expand(self.sizes[0], -1)
        ones = torch.ones(sizes + [1])
        self.grid = torch.cat([i_vals.unsqueeze(-1), j_vals.unsqueeze(-1), ones], axis=-1) + 0.02 * torch.randn(np.prod(self.grid_shape)).reshape(self.grid_shape)

        self.K = 10
        self.space_dim = space_dim
        self.sigma1s = []
        self.sigma2s = []
        self.epss = []
        self.score_history = []
        self.validation_score_history = []

    def reset(self):
        self.gamma = 0
        self.s = 0

    def forward(self, x, squeeze=True):
        #print(x.shape, self.grid.shape)
        X = x.view([-1, 1, 1, self.space_dim])
        d = torch.sum((self.grid - X)**2, axis=-1) ** (.5)
        if squeeze:
            d = d.squeeze()
        return d


    def fit(self, x : torch.Tensor, y : torch.Tensor, epochs : int=1, disable_tqdm : bool=False, x_valid : torch.Tensor=None) -> 'AniSOM':
        """ Fit the model

        :param torch.Tensor x: the embedded x time series data, this is the variable that goes into the predictor in the end
        :param torch.Tensor y: the embedded y data, that guides the training through the neighborhoods
        :param int epochs: number of full cycles of training (all data point skimmed through 1 cycle)
        :param bool didable_tqdm: Not to show progress bar of the training
        :param torch.Tensor x_valid: (optional) validation set embedded x time_series to calculate validation score during training
        :return: self
        :rtype: AniSOM
        """
        X = x
        Y = y
        N = x.shape[0]
        total = N * epochs
        i_vals = torch.arange(self.sizes[0]).view([-1, 1]).expand(-1, self.sizes[1])
        j_vals = torch.arange(self.sizes[1]).view([1, -1]).expand(self.sizes[0], -1)
        G = self.grid

        steps = 0
        pbar = tqdm(total=total, disable=disable_tqdm)
        for epoch in range(epochs):
            for s in range(N):
                # compute the actual values of the learning rate
                sigma1 = 10 * np.exp( - steps / total)
                sigma2 = 20 * np.exp( - steps * np.log(5) / total)
                eps = 0.2 * np.exp( - steps * np.log(20) / total)

                # pick a random point
                ts = np.random.randint(low=0, high=N, size=1)

                # Compute the activations
                d = self.forward(X[ts])

                # Find the best matching unit
                ij = torch.where(d == torch.min(d))
                i_star, j_star = ij[0][0], ij[1][0]

                # Find the neighborhood but in y
                dists, inds_ts = cKDTree(Y).query(Y[ts], k=self.K + 1)

                inds_ts = inds_ts[0, 1:]
                for k in range(self.K):
                    # print("This line:", inds_ts[k], j_star, G.shape)
                    d2 = self.forward(X[inds_ts[k]])[:, j_star:j_star+1]
                    ia = torch.where(d2 == torch.min(d2))[0][0]

                    w = torch.exp( - (i_vals - ia) ** 2 / sigma1 - (j_vals - j_star) ** 2 / sigma2 )

                    G += eps * w.view(self.sizes + [1]).expand([-1, -1, self.space_dim]) * ( X[inds_ts[k]].view([1, 1, self.space_dim]).expand(self.sizes + [-1]) - G)

                steps += 1
                pbar.update(1)
                self.sigma1s.append(sigma1)
                self.sigma2s.append(sigma2)
                self.epss.append(eps)

                if s % 100 == 0:
                    # compute the score
                    score = self.compute_score(X)
                    self.score_history.append(score)
                    if x_valid is not None:
                        validation_score = self.compute_score(x_valid)
                        self.validation_score_history.append(validation_score)

            self.grid = G

        return self

    def predict(self, X):
        def get_coords(activations):
            ij_argmin = (activations == torch.min(activations)).nonzero()
            if ij_argmin.shape[0] > 1:
                ij_argmin = ij_argmin[:1]
            return ij_argmin

        activations = self.forward(X, squeeze=False)
        # print("The shape of activations is:", activations.shape)
        a = activations.shape[0]
        # print(a)
        ij_argmin = torch.cat([get_coords(activations[o]) for o in range(a)], axis=0)
        return ij_argmin
    
    def compute_score(self, X):
        """ Compute the mean minimum of activation ona  bunch of samples X.
        :param X: input samples
        :return: mean minimum of activation 
        """
        return self.forward(X).min(axis=-1).values.min(axis=-1).values.mean()

        
