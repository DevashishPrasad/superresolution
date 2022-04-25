import torch
from torch import nn
from torchsummary import summary

from data import *
from model import *
from train import *

import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('./config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Config parameters
train_hr = config['data_paths']['train_dir']
valid_hr = config['data_paths']['valid_dir']
degradation_params = config['degradation_params']

mini_b_size = config['training_params']['mini_batch_size']
lr = config['training_params']['learning_rate']
epochs = config['training_params']['epochs']
patience = config['training_params']['patience']


train_dataset = Div2kDataset(train_hr, degradation_params, 'train')
val_dataset = Div2kDataset(valid_hr, degradation_params, 'val')

trainloader = DataLoader(train_dataset, batch_size=mini_b_size, shuffle=True)
validloader = DataLoader(val_dataset, batch_size=mini_b_size, shuffle=True)


dataloaders = {'train': trainloader, 'val': validloader}

## Create model
model = edsr().to(device)
params_to_update = model.parameters()
summary(model, (3, 64, 64), batch_size=mini_b_size)

## Initialize optimizers, schedulers and criterion
optimizer_ft = optim.Adam(params_to_update, lr=lr, betas=(0.9, 0.999))
decayRate = 0.5
lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.1, threshold=0.01, min_lr=0.000001, patience=patience)
criterion = nn.L1Loss()

## Execute Training
train_model(model, dataloaders, criterion, optimizer_ft, lr_decay, mini_batch_size=mini_b_size, num_epochs=epochs)
