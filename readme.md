<!-- # aniSOM

aniSOM is a Python implementation of the Anisotropic Self-Organizing Map (ASOM), a neural network model for reconstruction of shared hidden driver between time series. This project is tailored for experiments with time series and dynamical systems, such as the logistic map.
 -->
# aniSOM

aniSOM is a Python implementation of the Anisotropic Self-Organizing Map (ASOM), a neural network model for reconstruction of shared hidden driver between time series.

## Installation
Download the source code:
```bash
$git clone git@github.com:tncslab/aniSOM.git
```

Step into the directory:
```bash
$cd anisom
```

Create a new conda environment and install dependencies:

```bash
$conda create -n anisom
$conda activate anisom
(anisom) $conda install python=3.11

```

Install the package:
```bash
(anisom) $pip install -e .
```

## Usage

Below is a minimal example showing how to preprocess data and fit the model:

```python
import numpy as np
import torch
from anisom.network.anisom import AniSOM
from anisom.datagen.logmap import gen_logmapdata
from anisom.preprocessing.tde import time_delay_embedding

# Generate synthetic data
dataset, _ = gen_logmapdata(dict(N=1, n=100, rint=(3.8, 4.), 
                                 A0=np.array([[0,0,0],[1,0,0],[1,0,0]]), 
                                 A=np.array([[1.,0.,0.],[0.3,1.,0.],[0.4,0.,1.]])))
x = dataset[0][:, 1]
y = dataset[0][:, 2]

# Time-delay embedding
X = torch.tensor(time_delay_embedding(x, dimension=3, delay=1))
Y = torch.tensor(time_delay_embedding(y, dimension=3, delay=1))

# Initialize and fit AniSOM
model = AniSOM(space_dim=3, grid_dim=2, sizes=[40, 20])
model.fit(X, Y, epochs=2)
```

See the `examples/` folder for a more detailed examples.