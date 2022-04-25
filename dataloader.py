import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import math
import os
import glob
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

import datetime

listener_AU_dir = "./listener"
audio_feats_dir = "./audio_feats"


class RLD_dataset(Dataset):

    def __init__(self, listener_AU_dir, audio_feats_dir):
        # openface feats for listener
        listener_AU_files = os.path.join(listener_AU_dir, "*.csv")
        listener_AU = glob.glob(listener_AU_files)
        listener_AU = pd.concat(map(pd.read_csv, listener_AU), ignore_index= True)
        first_column = listener_AU.columns[0]
        listener_AU = listener_AU.drop([first_column], axis=1)
        listener_AUs = listener_AU.iloc[:-2,295:315]
        listener_AUs = listener_AUs.dropna()
        check_nan = listener_AUs.loc[pd.isna(listener_AUs).any(1), :].index
        n = list(listener_AUs.columns)
        print(n)
        print(list(check_nan))

        self.listner_AUs = torch.tensor(listener_AUs.values)

        print(len(self.listner_AUs))

        #speech feats
        wav_dir = os.path.join(audio_feats_dir, "*.npy")
        wav_files  = sorted(glob.glob(wav_dir))
        arr = []
        count = 0
        for f in wav_files:
            # print(np.load(f))

            # arr.extend([np.zeros(45)])
            arr.extend([np.zeros(45)])
            arr.extend([np.zeros(45)])
            arr.extend([np.zeros(45)])
            arr.extend([np.zeros(45)])
            # arr.append([np.zeros(45)])
            arr.extend(np.load(f))
            count = count + 1
            # if count %2 == 0:
                # arr.extend([np.zeros(45)])
            # print(arr)
        # normalized = []
        # for a in arr:
        #     n = (a - min(a)) / (max(a) - min(a))
        #     if np.isnan(n).at = 10
        #     ny() == True:
        #         print("y")
        #     normalized.append(n)
            # print(arr)

        # speech_feats = np.array(arr)
        self.speech_feats = torch.tensor(arr)
        print(len(self.speech_feats[0]))


    def train_val_test_split(self, test_ratio):
        val_ratio = test_ratio / (1 - test_ratio)
        X_train, X_test, y_train, y_test = train_test_split(self.speech_feats, self.listner_AUs, test_size=test_ratio, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(self.speech_feats, self.listner_AUs, test_size=val_ratio, shuffle=False)
        train = TensorDataset(X_train, y_train)
        val = TensorDataset(X_val, y_val)
        test = TensorDataset(X_test, y_test)
        return train, val, test

    # def __getitem__(self, index):
    #     return self.train
    #
    # def __len__(self):
    #     returnn self.train.shape[0]


# rld = RLD_dataset(listener_AU_dir,audio_feats_dir)
# train, val, test = rld.tensordataset()
# train_loader = DataLoader(dataset=train, batch_size=64, shuffle=False, drop_last=True)
# val_loader = DataLoader(dataset=val, batch_size=64, shuffle=False, drop_last=True)
# test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False, drop_last=True)
# # train = TensorDataset(train_features, train_targets)
# # dataiter = iter(dataloader)
# # data = dataiter.next()
# # feature, label = data
# input_dim = 45
# output_dim =22
# hidden_dim = 90
# layer_dim = 3
# batch_size = 64
# dropout = 0.2
# n_epochs = 100
# learning_rate = 1e-3
# weight_decay = 1e-6
# model_params = {'input_dim': input_dim,
#                 'hidden_dim' : hidden_dim,
#                 'layer_dim' : layer_dim,
#                 'output_dim' : output_dim,
#                 'dropout_prob' : dropout}
#
# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout)
#
# loss_fn = nn.MSELoss(reduction='mean')
# optimizer = optim.Adam(model.parameters(), lr=1e-2)
#
# train_losses = []
# # val_losses = []
# # train_step = make_train_step(model, criterion, optimizer)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
# opt.train(train_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
# opt.plot_losses()
#
# predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)
#
#
# # for epoch in range(n_epochs):
# #     batch_losses = []
# #     for x_batch, y_batch in train_loader:
# #         x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
# #         y_batch = y_batch.to(device)
# #         loss = train_step(x_batch, y_batch)
# #         batch_losses.append(loss)
# #     training_loss = np.mean(batch_losses)
# #     train_losses.append(training_loss)
# #     # with torch.no_grad():
# #     #     batch_val_losses = []
# #     #     for x_val, y_val in val_loader:
# #     #         x_val = x_val.view([batch_size, -1, n_features]).to(device)
# #     #         y_val = y_val.to(device)
# #     #         model.eval()
# #     #         yhat = model(x_val)
# #     #         val_loss = criterion(y_val, yhat).item()
# #     #         batch_val_losses.append(val_loss)
# #     #     validation_loss = np.mean(batch_val_losses)
# #     #     val_losses.append(validation_loss)
# #
#     print(f"[{epoch + 1}] Training loss: {training_loss:.4f}")