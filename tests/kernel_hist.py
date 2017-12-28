import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm, trange
from ipywidgets import interact, fixed

sns.set_context("poster")
sns.set_style("ticks")

%matplotlib inline

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
T = iris.target

X.shape[0]

kf = KFold(n_splits=15, shuffle=False)

for train_idx, val_idx in kf.split(X=X, y=T):
  train_x, val_x = X[train_idx], X[val_idx]
  train_t, val_t = T[train_idx], T[val_idx]
  print(val_idx)




N = 100
min_x, max_x = -30, 30
(X_obs, y_obs, X_true, y_true), (w, b, true_model) = get_data(N, min_x, max_x)

plt.plot(X_obs, y_obs, ls="none", marker="o", color="k", label="observed")
plt.plot(X_true, y_true, ls="-", color="r", label="true")
plt.legend()
sns.despine(offset=10)

x = np.random.randn(5,3,2)

x1 = torch.Tensor([[1,1],[1,1],[1,1]])
x1
x2 = torch.Tensor([[2,2],[2,2],[2,2]])
x2

y = torch.cat([x1,x2,x1,x2], dim=1)
y
y1 = torch.split(y, 3, dim=1)
y1

x = Variable(torch.randn(10,3,2).cuda())
x.size()
x.transpose(1,2).size()


torch.stack([x1, x1, x2, x2], dim=1)[:,3,:]
torch.stack([x1, x2], dim=1)



x = torch.cuda.FloatTensor([[2, 3],[4, 1]])
x
x.cpu()


torch.arange(2, 5, 0.5)
torch.linspace(2, 5, 6)
