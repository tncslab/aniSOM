import numpy as np
from tqdm.auto import tqdm
from types import SimpleNamespace


class LogMap:
    """
    Simulates a multi-dimensional logistic map system.

    The logistic map is a classic example of how complex, chaotic behaviour
    can arise from very simple non-linear dynamical equations. This class
    extends it to multiple dimensions with coupling.
    """
    def __init__(self, r: np.ndarray, A: np.ndarray, x0: np.ndarray):
        """
        Initializes the LogMap system.

        :param np.ndarray r: Growth rate parameters for each dimension.
        :param np.ndarray A: The coupling matrix between dimensions.
        :param np.ndarray x0: Initial state vector for the system.
        """
        self.r = r
        self.A = A
        self.x0 = x0

    def bounder(self, x: np.ndarray) -> np.ndarray:
        """Applies boundary condition on data"""
        z = np.array(x)
        while not np.all(np.logical_and(z <= 1, z >= 0)):
            z = np.abs(z)
            inds = np.where(z > 1)[0]
            for i in inds:
                z[i] = 1 - (z[i] - 1)
        return z

    def step(self, x : np.ndarray) -> np.ndarray:
        return self.bounder(self.r * x * (np.ones(len(x)) - self.A @ x))

    def gen_dataset(self, n : int) -> np.ndarray:
        x = [self.x0]
        for i in range(n - 1):
            x.append(self.step(x[-1]))
        return np.array(x)


class LogmapExpRunner:
    """
    Manages the generation of datasets from the LogMap system by sampling
    parameters and running simulations.

    This class facilitates running multiple experiments with varying parameters
    for the LogMap.
    """
    def __init__(self, nvars : int, baseA : np.ndarray, r_interval: list):
        """
        Initializes the LogmapExpRunner.

        :param int nvars: The number of variables (dimensions) in the LogMap system.
        :param np.ndarray baseA: A base coupling matrix. This matrix is scaled
                                 randomly when sampling parameters.
        :param list r_interval: A list or tuple `[min_r, max_r]` specifying the
                                interval from which to sample the 'r' growth
                                rate parameters.
        """
        self.nvars = nvars
        self.baseA = baseA
        self.r_interval = r_interval

    def sample_params(self, A=None, r=None, x0=None, seed=None):
        """
        Samples parameters for a LogMap simulation.

        If parameters `A`, `r`, or `x0` are provided, they are used directly.
        Otherwise, they are sampled randomly based on the runner's configuration.

        :param np.ndarray A: (Optional) The coupling matrix. If None, a new
                             matrix is sampled.
        :param np.ndarray r: (Optional) The growth rate parameters. If None, new
                             parameters are sampled from `r_interval`.
        :param np.ndarray x0: (Optional) The initial state vector. If None, a new
                              random state is sampled.
        :param int seed: (Optional) Seed for the random number generator for
                         reproducibility.
        :return: A tuple containing the sampled (or provided) `r`, `A`, and `x0`.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        np.random.seed(seed)
        rint = self.r_interval
        if r is None:
            r = rint[0] + (rint[1] - rint[0]) * np.random.rand(self.nvars)

        if A is None:
            A = self.baseA * (0.4 * np.random.rand(self.nvars ** 2).reshape([self.nvars, self.nvars]) + 0.1)
        np.fill_diagonal(A, 1)

        if x0 is None:
            x0 = np.random.rand(self.nvars)
        self.A = A
        self.r = r
        self.x0 = x0
        return r, A, x0

    def gen_experiment(self, 
                       n : int, 
                       A : np.ndarray = None ,
                       r : np.ndarray = None,
                       x0 : np.ndarray = None,
                       seed: int = None) -> tuple[np.ndarray, dict]:
        """
        Generates a single dataset (experiment) using the LogMap system.

        It first samples or uses provided parameters, then runs the LogMap
        simulation to generate time series data.

        :param int n: The number of time steps (length of the dataset) to generate.
        :param np.ndarray A: (Optional) The coupling matrix. If None, it's sampled.
        :param np.ndarray r: (Optional) The growth rate parameters. If None, they are
                             sampled.
        :param np.ndarray x0: (Optional) The initial state vector. If None, it's
                              sampled.
        :param int seed: (Optional) Seed for the random number generator.
        :return: A tuple containing the generated dataset and a dictionary of the
                 parameters used for the simulation.
        :rtype: tuple[np.ndarray, dict]
        """
        r, A, x0 = self.sample_params(r=r, A=A, x0=x0, seed=seed)
        data = LogMap(r=r, A=A, x0=x0).gen_dataset(n)
        self.data = data
        return data, {'r': r, 'A': A, 'x0': x0}


def gen_logmapdata(param_dict: dict) -> tuple[list[np.ndarray], list[dict]]:
    """Generate logmap dataset

    :param dict param_dict: dict containing the parameters for the logmap dataset
    :return: (datasets, parameters) tuple of data and parameters
    :rtype: tuple
    """
    dparams = SimpleNamespace(**param_dict)
    N = dparams.N  # number of realizations
    n = dparams.n  # Length of time series
    rint = dparams.rint  # interval to chose from the value of r parameter
    A0 = dparams.A0
    A = dparams.A

    dataset, params = zip(*[LogmapExpRunner(nvars=3,
                                            baseA=A0,
                                            r_interval=rint).gen_experiment(n=n,
                                                                            A=A,
                                                                            seed=i) for i in
                            tqdm(range(N), desc='Generating dataset')])
    return dataset, params
