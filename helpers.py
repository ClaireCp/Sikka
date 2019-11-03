import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad

import os
import numpy as np
import librosa
import soundfile as sf

def standardize(x):
    mean_x = x.mean()
    x = x - mean_x
    std_x = x.std()
    x = x / std_x
    return x, mean_x, std_x

def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

def shift(data):
    return np.roll(data, 1600)

def stretch(data, input_length, rate=1):
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def get_data(n_mfcc, max_signal_len, max_pad_len, lim):
    n_features = 400
    feature_dim = 20*n_mfcc
    input = np.zeros((n_features, feature_dim))
    target = np.zeros((n_features))
    test_indices = np.zeros((n_features))

    for i, filename  in enumerate(os.listdir('data')):
        data, sr = sf.read('data/'+filename)
        mfcc_feat = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfcc_feat.shape[1]
        mfcc_feat = np.pad(mfcc_feat, pad_width=((0, 0), (0, pad_width)), mode='constant')
        input[i] = mfcc_feat.reshape((-1))
        target[i] = int(filename[0])

        if int(filename.split('.')[0][-1]) < lim:
            test_indices[i] = 1
            
    return input, target, test_indices

def augmented_data(n_mfcc, max_signal_len, max_pad_len, lim):
    n_features = (10-lim)*40*5 + lim*40
    feature_dim = 20*n_mfcc
    input = np.zeros((n_features, feature_dim))
    target = np.zeros((n_features))
    test_indices = np.zeros((n_features))
    
    k = 0
    for i, filename  in enumerate(os.listdir('data')):
        data, sr = sf.read('data/'+filename)

        if int(filename.split('.')[0][-1]) >= lim:
            data_ls = [data, stretch(data, max_signal_len, 0.95), stretch(data, max_signal_len, 1.05), \
                       shift(data), add_noise(data)]

            for j, data in enumerate(data_ls):
                mfcc_feat = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
                pad_width = max_pad_len - mfcc_feat.shape[1]
                mfcc_feat = np.pad(mfcc_feat, pad_width=((0, 0), (0, pad_width)), mode='constant')
                input[k] = mfcc_feat.reshape((-1))
                target[k] = int(filename[0])
                k += 1

        else:
            test_indices[k] = 1
            mfcc_feat = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
            pad_width = max_pad_len - mfcc_feat.shape[1]
            mfcc_feat = np.pad(mfcc_feat, pad_width=((0, 0), (0, pad_width)), mode='constant')
            input[k] = mfcc_feat.reshape((-1))
            target[k] = int(filename[0])
            k += 1
    
    return input, target, test_indices

""" ************************************************************************************************ """

def train_ce(model, train_input, train_target, test_input, test_target, optimizer, nb_epochs=300, batch_size=100):
    criterion = torch.nn.CrossEntropyLoss()
    
    nb_samples = train_input.size(0)
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    
    for epoch in range(nb_epochs):
        running_train_corrects = 0
        running_train_loss = 0.0
        
        for b in range(0, nb_samples, batch_size):
            optimizer.zero_grad()
            train_output = model(train_input.narrow(0, b, batch_size))
            loss = criterion(train_output, train_target.narrow(0, b, batch_size))
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * nb_samples
            pred_train = torch.max(train_output, 1)[1]
            running_train_corrects += torch.sum(pred_train == train_target.narrow(0, b, batch_size))
            
        test_output = model(test_input)
        test_loss = criterion(test_output, test_target)
        pred_test = torch.max(test_output, 1)[1]
        test_corrects = torch.sum(pred_test == test_target)
        
        if epoch % 50 == 0:
            epoch_train_loss = running_train_loss / nb_samples
            epoch_test_loss = test_loss.item() / test_input.size(0)
            train_loss_history.append(epoch_train_loss)
            test_loss_history.append(epoch_test_loss)
            epoch_train_acc = running_train_corrects.double() / nb_samples
            train_acc_history.append(epoch_train_acc)
            epoch_test_acc = test_corrects.double() / test_input.size(0)
            test_acc_history.append(epoch_test_acc)
            
        if epoch % 50 == 0:
            print('epoch {}, train loss {}, test loss {}, train acc {}, test acc {}'
                  .format(epoch, round(epoch_train_loss, 4), round(epoch_test_loss, 4), epoch_train_acc, epoch_test_acc))
        
    return train_loss_history, test_loss_history, train_acc_history, test_acc_history