import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataloader import RLD_dataset
from model import LSTMModel
from optimization import Optimization


listener_AU_dir = "./listener"
audio_feats_dir = "./audio_feats"
rld = RLD_dataset(listener_AU_dir,audio_feats_dir)
train, val, test = rld.train_val_test_split(0.2)
train_loader = DataLoader(dataset=train, batch_size=64, shuffle=False, drop_last=True)
val_loader = DataLoader(dataset=val, batch_size=64, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset=test, batch_size=64, shuffle=False, drop_last=True)
test_loader_one = DataLoader(dataset=test, batch_size=1, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_dim = 45
output_dim = 20
hidden_dim = 90
layer_dim = 3
batch_size = 64
dropout = 0.2
n_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-6

# # model_params = {'input_dim': input_dim,
# #                 'hidden_dim' : hidden_dim,
# #                 'layer_dim' : layer_dim,
# #                 'output_dim' : output_dim,
# #                 'dropout_prob' : dropout}
# #
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout)
#
loss_fn = nn.MSELoss(reduction= "mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#
opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()
#
predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)