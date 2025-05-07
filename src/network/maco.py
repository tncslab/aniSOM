import numpy as np
import torch
from torch.nn import Module, Sequential, Linear, ReLU, MSELoss, PReLU
from torch import Tensor
from torch.nn.functional import relu
import torch.optim as optim
from torchvision import transforms

from tqdm import tqdm
from functools import partial
from cdriver.preprocessing.tde import TimeDelayEmbeddingTransform, cropper

from scipy.signal import correlate, correlation_lags
from scipy.stats import rankdata

from sklearn.preprocessing import scale


def get_mapper(n_in, n_h1, n_h2, n_out):
    mapper = Sequential(Linear(n_in, n_h1),
                        ReLU(),
                        Linear(n_h1, n_h2),
                        ReLU(),
                        Linear(n_h2, n_out))
    return mapper


def get_coach(n_in, n_h1, n_out):
    coach = Sequential(Linear(n_in, n_h1),
                       ReLU(),
                       Linear(n_h1, n_out))
    return coach


class MaCo(torch.nn.Module):
    def __init__(self, Ex, Ey, Ez, mh_kwargs, ch_kwargs, preprocess_kwargs, device, c=0):
        super().__init__()

        self.mapper = get_mapper(n_in=Ey, n_out=Ez, **mh_kwargs)
        self.coach_x = get_coach(Ex + Ez, **ch_kwargs)

        self.preprocess_params = preprocess_kwargs
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.mh_params = mh_kwargs
        self.ch_params = ch_kwargs
        self.train_loss_history = []
        self.criterion = MSELoss()
        self.device = device
        self.c = c  # regularization parameter for the loss function
        self.mapper.to(device)
        self.coach_x.to(device)
        self.to(device)

    def preprocess(self, x, y):
        """Preprocess the input data

        :param q: input data
        :return: preprocessed data
        """
        # print(type(x), type(y))
        d_embed_x = self.Ex + 1  # embedding for the variable to be predicted, the additional dim is for the target
        d_embed_y = self.Ey  # embedding for the other variable
        tau = self.preprocess_params['tau']  # embedding delay

        # Crop to ensure the same length of the time series (and also causality)
        input_crop = (d_embed_y - d_embed_x) * tau

        # Define transforms
        common_transform = transforms.Compose([transforms.Normalize((0.5), (.3)),
                                               torch.Tensor.float,
                                               partial(torch.squeeze, axis=0)])

        y_transform = transforms.Compose([common_transform,
                                          TimeDelayEmbeddingTransform(d_embed_y, tau),
                                          partial(cropper, location='first', n=-input_crop)])

        xt_transform = transforms.Compose([common_transform,
                                          TimeDelayEmbeddingTransform(d_embed_x, tau),
                                          partial(cropper, location='first', n=input_crop)])

        y_transformed = y_transform(y)
        xt_transformed = xt_transform(x)
        x_transformed = xt_transformed[:, :-1]
        target_transformed = xt_transformed[:, -1:]

        # print("shapes after tansforms:", x_transformed.shape, target_transformed.shape, y_transformed.shape)
        return x_transformed.to(self.device), target_transformed.to(self.device), y_transformed.to(self.device)

    def forward(self, x, y):
        # print("Device of x before preprocessing:", x.device)
        # print("Device of y before preprocessing:", y.device)
        X, target, Y = self.preprocess(x, y)
        # print("Device of X in forward mapper:", X.device)
        # print("Device of Y in forward mapper:", Y.device)
        # print("Device of target in forward mapper:", target.device)

        # print("shape of Y in forward mapper:", Y.shape)
        z = self.mapper.forward(Y)
        hz = relu(z)

        mx = torch.cat([hz, X], axis=1)

        pred = self.coach_x.forward(mx)
        return pred, z, hz, target

    def regularized_loss(self, target, z, pred):
        """Loss function to incorporate the Frobenius norm of the correlation matrix

        :param q:
        :param z:
        :param pred:
        :return:
        """
        E = MSELoss()(target, pred)
        # E = MSELoss()(q, pred) - self.c * torch.linalg.det(torch.corrcoef(z.T)) + self.c * torch.norm(torch.corrcoef(z.T))
        # E = MSELoss()(q, pred) - self.c * torch.linalg.det(torch.corrcoef(z.T))
        # E = MSELoss()(q, pred) + self.c * torch.norm(torch.corrcoef(z.T))

        return E
    def train_loop(self, loader, n_epochs, lr=1e-2, disable_tqdm=False):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        loss_hist = self.train_loss_history.copy()
        for epoch in tqdm(range(n_epochs), leave=False, disable=disable_tqdm):
            # print("epoch {}.".format(epoch))
            losses = []
            for i, batch in enumerate(loader):
                x, y = batch

                self.optimizer.zero_grad()

                pred, z, hz, target = self.forward(x.to(self.device), y.to(self.device))

                # print(target.is_cuda, pred.is_cuda, z.is_cuda)
                loss = self.regularized_loss(target, z, pred)

                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            loss_hist.append(np.mean(losses))
        self.train_loss_history = loss_hist.copy()
        return loss_hist

    def test_loop(self, loader):
        with torch.no_grad():
            x, y = loader
            pred, z, hz, target = self.forward(x.to(self.device), y.to(self.device))
            loss = self.criterion(target, pred).item()
        return loss

    def valid_loop(self, loader):
        x, y = loader
        pred, z, hz, target = self.forward(x.to(self.device), y.to(self.device))
        loss = self.criterion(target, pred).item()
        return loss, pred.squeeze().detach().cpu().numpy(), z.squeeze().detach().cpu().numpy(), hz.squeeze().detach().cpu().numpy()
