import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.utils.data #データセット読み込み関連
import torchvision #画像関連
import numpy as np
from get_data import get_data
from sklearn.utils import shuffle
import torch.cuda
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.pardir)
from mymodule.trainer import Trainer
from mymodule.utils import data_loader, evaluator
from tensorboardX import SummaryWriter
from mymodule.layers import BayesLSTM, Bayes_classifier

writer = SummaryWriter()
in_size = 28
hidden_size = 100
batch_size = 1024
linear_hidden_size = 512
epochs = 1000
sample = 10
name = 'BayesLSTM_y'
'''
データの生成
'''
train, test, label = get_data(idx=1)
train = train.reshape(-1, 28, 28)
test = test.reshape(-1, 28, 28)
train_loader = data_loader(train, label, batch_size=batch_size,
                           shuffle=True, gpu=False)
test_loader = data_loader(test, label, batch_size=batch_size,
                           shuffle=False, gpu=False)

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
            in_dropout=0.6,
            hidden_dropout=0.6,
            out_dropout=0.6,
            linear_in_size=hidden_size,
            linear_hidden_size=linear_hidden_size,
            linear_out_size=2,
            linear_dropout=0.5,
            gpu=True)
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=.001)

# out = model(torch.autograd.Variable(
#       torch.cuda.FloatTensor(batch_size, train.shape[1],
#                              train.shape[2]),
#                              requires_grad=True))
# writer.add_graph(model, out)


trainer = Trainer(model, criterion, optimizer,
                  train_loader, test_loader,
                  val_num=1, early_stopping=None,
                  writer=writer)
trainer.run(epochs=epochs)

torch.save(model.state_dict(), name + '.pth')

bayes_predictor = Bayes_classifier(predictor=model, num_sample=sample)
loss, acc, pre_array = evaluator(bayes_predictor, criterion, test_loader)
print('Test Accuracy of the model on {} test data:{:0.4f}'.format(
      test.shape[0] , acc))


dt = 1/128
N = test.shape[0]
t = np.linspace(1, N, N) * dt - dt

plt.plot(t, label)
plt.plot(t, pre_array[:,1])
plt.grid()
plt.xlabel('time')
plt.ylabel('predict and label')
plt.title('test infer')
plt.show()
