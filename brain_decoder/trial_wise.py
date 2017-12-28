import os, sys
sys.path.append(os.pardir)
import numpy as np
from numpy.random import RandomState
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import mne
from mne.io import concatenate_raws
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.torch_ext.util import set_random_seeds
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
from mymodule.utils import data_loader, evaluator
from mymodule.layers import LSTM, Residual_block, Res_net
from mymodule.trainer import Trainer
from mymodule.optim import Eve, YFOptimizer
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter

#
#
# # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
# subject_id = 1
# event_codes = [5,6,9,10,13,14]
#
# # This will download the files if you don't have them yet,
# # and then return the paths to the files.
# physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)
#
# # Load each of the files
# parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto', verbose='WARNING')
#          for path in physionet_paths]
#
# # Concatenate them
# raw = concatenate_raws(parts)
#
# # bandpass filter
# raw.filter(5., 25., fir_design='firwin', skip_by_annotation='edge')
#
# # Find the events in this dataset
# events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
#
# # Use only EEG channels
# eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
#                    exclude='bads')
#
# # Extract trials, only using EEG channels
# epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=0, tmax=4.1, proj=False, picks=eeg_channel_inds,
#                 baseline=None, preload=True)
# # change time length
# # epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
#
#
# # Convert data from volt to millivolt
# # Pytorch expects float32 for input and int64 for labels.
#
# train_X = (epoched.get_data() * 1e6).astype(np.float32)
# train_y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1

physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,[4,8,12,]) for sub_id in range(1,10)]
physionet_paths = np.concatenate(physionet_paths)
parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
         for path in physionet_paths]

raw = concatenate_raws(parts)
# raw.filter(5., 35., fir_design='firwin', skip_by_annotation='edge')

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
events
# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=1, tmax=4.1, proj=False, picks=picks,
                baseline=None, preload=True)


physionet_paths_test = [mne.datasets.eegbci.load_data(sub_id,[4,8,12,]) for sub_id in range(51,56)]
physionet_paths_test = np.concatenate(physionet_paths_test)
parts_test = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
         for path in physionet_paths_test]
raw_test = concatenate_raws(parts_test)
# raw_test.filter(5., 35., fir_design='firwin', skip_by_annotation='edge')


picks_test = mne.pick_types(raw_test.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

events_test = mne.find_events(raw_test, shortest_event=0, stim_channel='STI 014')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epoched_test = mne.Epochs(raw_test, events_test, dict(hands=2, feet=3), tmin=-.5, tmax=3.0, proj=False, picks=picks_test,
                baseline=None, preload=True)

train_X = (epoched.get_data() * 1e6).astype(np.float32)
train_y = (epoched.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1
test_X = (epoched_test.get_data() * 1e6).astype(np.float32)
test_y = (epoched_test.events[:,2] - 2).astype(np.int64) #2,3 -> 0,1



# cut = int(np.ceil(train_X.shape[0]*(6/7)))
# train_ = train_X[:cut]
# val_ = train_X[cut:]
# train_t = train_y[:cut]
# val_t = train_y[cut:]

f_dim = train_X.shape[1]
seq_len = train_X.shape[2]
'''
モデル
'''
epochs = 1000
batch_size = 16
lr = 1e-5
train_loader = data_loader(train_X, train_y, batch_size=batch_size,
                           shuffle=True, gpu=False)
# test_loader = data_loader(test_, test_t, batch_size=batch_size)
val_loader = data_loader(test_X, test_y, batch_size=batch_size)

### resnet
res_ch = [64, 128, 256]
pooling = [int(seq_len/2), int(seq_len/4), int(seq_len/8)]
res_dropout = 0.9

### lstm
lstm_units = [res_ch[-1], 32]
lstm_dropout = 0.9
bi = True

### linear
dense_dropout = 0.9
linear_units = [(bi+1) * lstm_units[-1], 128, 2]

### reguralization
l2_regulizer = 1e-1


model = nn.Sequential(
          Res_net(dropout=res_dropout, res_ch=res_ch),
          LSTM(num_layers=1,
               in_size=lstm_units[0],
               hidden_size=lstm_units[1],
               batch_size=batch_size,
               dropout=lstm_dropout,
               bidirectional=bi,
               return_seq=False,
               gpu=True,
               continue_seq=False),
          nn.Linear(linear_units[0], linear_units[1]),
          nn.BatchNorm1d(linear_units[1]),
          nn.Dropout(dense_dropout),
          nn.Linear(linear_units[1], linear_units[2]),
          )

model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             weight_decay=l2_regulizer,
                             lr = lr)
# optimizer = Eve(model.parameters(),
#                 weight_decay=l2_regulizer,
#                 lr = lr)

##for tensorboard
# r = model(Variable(torch.ones(train_X.shape).cuda()))
writer = SummaryWriter()
# writer.add_graph(model, r)

### train_loop
trainer = Trainer(model, criterion, optimizer,
                  train_loader, val_loader,
                  val_num=1, early_stopping=2,
                  writer=writer)
trainer.run(epochs=epochs)
