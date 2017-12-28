import numpy as np
import cupy as cp
import os, sys
sys.path.append(os.pardir)
from load_foot import Load_data, make_data
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from mymodule.layers import LSTM, T_CNN
from tensorboardX import SummaryWriter
from mymodule.trainer import Trainer
from mymodule.layers import BayesLSTM, Bayes_classifier
from mymodule.utils import data_loader, evaluator
from torch import nn
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
from sklearn.utils import shuffle
import time

writer = SummaryWriter()
in_size = 10
hidden_size = 100
linear_hidden_size = 512
name = 'BayesLSTM_y'
epochs = 10
seq_len = 128
batch_size = 128
sample = 100
### preprocess 'ica' or 'pca'
preprocess= None
whiten = True
### feed pin_memory
pin = True

'''
データ前処理
'''
os.chdir('data')
data = Load_data(train_mat='train_foot.mat',
                 test_mat='test_foot.mat',
                 train_label_mat='label_foot.mat',
                 test_label_mat='label_foot.mat')
os.chdir('..')

if preprocess=='pca':
  data.pca(whiten = whiten)
if preprocess=='ica':
  data.ica(whiten = whiten)
else:
  pass

# train_x, train_t, test_x, test_t = data.corr_data(seq_len)
train_x, train_t, test_x, test_t = data.get_data2d(seq_len)

'''
入力データ作成
'''
train_t = train_t[:,1]
test_t = test_t[:,1]

train_diff = np.zeros(train_x.shape)
test_diff = np.zeros(test_x.shape)
train_diff[:,1:seq_len,:] = np.diff(train_x, axis=1)
test_diff[:,1:seq_len,:] = np.diff(test_x, axis=1)

### (batch, seq, 10 = 5 + 5)
train_ = np.c_[train_x, train_diff]
test_ = np.c_[test_x, test_diff]

# train_ = train_.transpose(0,2,1)
# test_ = test_.transpose(0,2,1)

'''
validationとtrainingの分割
'''

train_x, train_tt = shuffle(train_, train_t)
cut = int(np.ceil(train_.shape[0]*(2/7)))
print(cut)
train_ = train_x[:cut]
val_ = train_x[cut:]
train_t = train_tt[:cut]
val_t = train_tt[cut:]

print(val_.shape)
print(train_.shape)

train_loader = data_loader(train_, train_t, batch_size=batch_size,
                           shuffle=True, gpu=False, pin_memory=pin)
test_loader = data_loader(test_, test_t, batch_size=2048, pin_memory=pin)
val_loader = data_loader(val_, val_t, batch_size=batch_size, pin_memory=pin)


class MLP(nn.Module):
  def __init__(self, in_size, hidden_size, batch_size,
               in_dropout, hidden_dropout, out_dropout,
               linear_in_size, linear_hidden_size, linear_out_size,
               linear_dropout, gpu=True):
    super(MLP, self).__init__()
    self.lstm = BayesLSTM(in_size=in_size,
                          hidden_size=hidden_size,
                          batch_size=batch_size,
                          in_dropout=in_dropout,
                          hidden_dropout=hidden_dropout,
                          out_dropout=out_dropout,
                          bias=True, gpu=gpu)
    self.l1 = nn.Linear(linear_in_size, linear_hidden_size)
    self.l2 = nn.Linear(linear_hidden_size, linear_out_size)
    self.linear_dropout = linear_dropout


  def forward(self, x):
    y_seq, _ = self.lstm(x)
    h = y_seq[:, -1, :]
    h = self.l1(h)
    h = F.dropout(h, p=self.linear_dropout, training=True)
    h = self.l2(h)
    return h

model = MLP(in_size=in_size,
            hidden_size=hidden_size,
            batch_size=batch_size,
            in_dropout=0.3,
            hidden_dropout=0.3,
            out_dropout=0.3,
            linear_in_size=hidden_size,
            linear_hidden_size=linear_hidden_size,
            linear_out_size=2,
            linear_dropout=0.5,
            gpu=True)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=.03)

# out = model(torch.autograd.Variable(
#       torch.cuda.FloatTensor(batch_size, train.shape[1],
#                              train.shape[2]),
#                              requires_grad=True))
# writer.add_graph(model, out)

start = time.time()
trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=None,
                  writer=writer)
trainer.run(epochs=epochs)
c_time = time.time() - start
print(c_time)

torch.save(model.state_dict(), name + '.pth')

bayes_predictor = Bayes_classifier(predictor=model, num_sample=sample)

loss, acc, pre_array = evaluator(bayes_predictor, criterion, test_loader)
print('Test Accuracy of the model on {} test data:{:0.4f}'.format(
      test_x.shape[0] , acc))


dt = 1/128
N = test_x.shape[0]
t = np.linspace(1, N, N) * dt - dt

plt.plot(t, test_t)
plt.plot(t, pre_array[:,1])
plt.grid()
plt.xlabel('time')
plt.ylabel('predict and label')
plt.title('test infer')
plt.show()
