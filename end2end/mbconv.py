import os, sys
sys.path.append(os.pardir)
from load_foot import Load_data, make_data
import matplotlib.pyplot as plt
import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
import mne
from mne.io import concatenate_raws
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from torch import nn
from torch import optim
from torch.autograd import Variable
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.torch_ext.util import np_to_var, var_to_np
import torch.nn.functional as F
from numpy.random import RandomState
import torch as th
from braindecode.experiments.monitors import compute_preds_per_trial_for_set


seq_len = 450
preprocess = 'ica'
whiten = False

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
train_y = train_t[:,1].astype(np.int64)
test_y = test_t[:,1].astype(np.int64)

train_diff = np.zeros(train_x.shape)
test_diff = np.zeros(test_x.shape)
train_diff[:,1:seq_len,:] = np.diff(train_x, axis=1)
test_diff[:,1:seq_len,:] = np.diff(test_x, axis=1)

### (batch, seq, 10 = 5 + 5)
train_ = np.c_[train_x, train_diff]
test_ = np.c_[test_x, test_diff]

train_X = train_.transpose(0,2,1).astype(np.float32)
test_X = test_.transpose(0,2,1).astype(np.float32)

train_set = SignalAndTarget(train_X, y=train_y)
test_set = SignalAndTarget(test_X, y=test_y)



# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = seq_len
n_classes = 2
in_chans = train_set.X.shape[1]
# final_conv_length determines the size of the receptive field of the ConvNet
model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length,
                        final_conv_length=12).create_network()
to_dense_prediction_model(model)

if cuda:
    model.cuda()




optimizer = optim.Adam(model.parameters())

# determine output size
test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
if cuda:
    test_input = test_input.cuda()
out = model(test_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]
print("{:d} predictions per input/trial".format(n_preds_per_input))

iterator = CropsFromTrialsIterator(batch_size=32,input_time_length=input_time_length,
                                  n_preds_per_input=n_preds_per_input)

rng = RandomState((2017,6,30))
for i_epoch in range(20):
    # Set model to training mode
    model.train()
    for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        outputs = model(net_in)
        # Mean predictions across trial
        # Note that this will give identical gradients to computing
        # a per-prediction loss (at least for the combination of log softmax activation
        # and negative log likelihood loss which we are using here)
        outputs = th.mean(outputs, dim=2, keepdim=False)
        loss = F.nll_loss(outputs, net_target)
        loss.backward()
        optimizer.step()

    # Print some statistics each epoch
    model.eval()
    print("Epoch {:d}".format(i_epoch))
    for setname, dataset in (('Train', train_set),('Test', test_set)):
        # Collect all predictions and losses
        all_preds = []
        all_losses = []
        batch_sizes = []
        for batch_X, batch_y in iterator.get_batches(dataset, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            all_preds.append(var_to_np(outputs))
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss = float(var_to_np(loss))
            all_losses.append(loss)
            batch_sizes.append(len(batch_X))
        # Compute mean per-input loss
        loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                       np.mean(batch_sizes))
        print("{:6s} Loss: {:.5f}".format(setname, loss))
        # Assign the predictions to the trials
        preds_per_trial = compute_preds_per_trial_for_set(all_preds,
                                                          input_time_length,
                                                          dataset)
        # preds per trial are now trials x classes x timesteps/predictions
        # Now mean across timesteps for each trial to get per-trial predictions
        meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
        predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
        accuracy = np.mean(predicted_labels == dataset.y)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))
