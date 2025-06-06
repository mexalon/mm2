import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import h5py
from tqdm.notebook import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import os
from pathlib import Path

CURR_DIR = os.getcwd()

# Dataset Class -------------------------------------------------------
class EventsToPermDataset(torch.utils.data.Dataset):

    def __init__(self, h5_path, transforms):
        super().__init__()
        self.path = h5_path
        self.transforms = self.stack_transforms(transforms) # transforms func
        with h5py.File(h5_path, 'a') as f:
            nmodels = f['events'].shape[0]
        self.nmodels = nmodels

    def __getitem__(self, idx):
        with h5py.File(self.path, 'a') as f:
            event = f['events'][idx].astype('float32')
            perm = f['perm'][idx].astype('float32')

        return self.transforms(event, perm)

    def __len__(self):
        return self.nmodels

    def stack_transforms(self, func_list):
        def performer(*args):
            for f in func_list:
                args = f(*args)
            return args
        return performer

class Normalizer:
    ''' returns log10(perm)     '''
    def __call__(self, events, perms):
        log_perm = torch.log10(perms)
        return events, log_perm

class ToTensor:
    '''Transforms numpy to torch tensors'''
    def __call__(self, events, perms):
        perms = torch.Tensor(perms)
        events = torch.Tensor(events)
        return events, perms

# Custom losses ---------------------------------------------------------

class MaskedL1Loss(nn.Module):
    # masked loss for better event handeling: no events - no results 
    def __init__(self, mask):
        super().__init__()
        self.base_loss = torch.nn.L1Loss(reduction='none')
        self.mask = mask

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        masked_loss = self.mask * self.base_loss(input, target)
        return torch.mean(masked_loss)
    
class MaskedMSELoss(nn.Module):
    # masked loss for better event handeling: no events - no results 
    def __init__(self, mask):
        super().__init__()
        self.base_loss = torch.nn.MSELoss(reduction='none')
        self.mask = mask

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        masked_loss = self.mask * self.base_loss(input, target)
        return torch.mean(masked_loss)

# model modules ---------------------------------------------------------

class EventListTransformerEncoder(nn.Module):
    '''
    Performs attention over the List of events
    '''
    def __init__(self, emb_size, num_layers, nheads):
        super().__init__()
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.nheads = nheads
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.emb_size, nhead = self.nheads, batch_first=True) # one head. emb size is unpredictable
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class Fc(nn.Module):
    '''
    universal Fc layer inp_size - > target size via num_layers
    '''
    def __init__(self, inp_size, target_size, num_layers, flat=False, fc_drop=0):
        super().__init__()
        self.target_size = target_size
        self.layers = self.create_layers(inp_size, target_size, num_layers, flat)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(fc_drop)

    def create_layers(self, inp, outp, nlayers, flat):
        if flat:
            sizes = [inp] + [outp for ii in range(nlayers)]
        else:
            mult = (outp/inp)**(1/nlayers)
            sizes = [int(inp*mult**ii) for ii in range(nlayers)]
            sizes.append(outp)

        layers = [nn.Linear(sizes[ii], sizes[ii+1]) for ii in range(nlayers)]
        return nn.ModuleList(layers)

    def forward(self, x):
        bs = x.shape[0] # batch size
        x = x.flatten(0, 1)
        for layer in self.layers[:-1]:
            x = self.drop(self.relu(layer(x)))
        x = self.layers[-1](x)
        x = x.view(bs, -1, self.target_size)
        return x
    
class Upconv(nn.Module):
    '''
    universal Fc layer inp_size - > target size via num_layers
    '''
    def __init__(self, event_num, small_cube_side, target_shape):
        super().__init__()
        self.small_cube_side = small_cube_side
        self.event_num = event_num
        self.target_shape = target_shape # no need now
        self.relu = nn.LeakyReLU(0.01)

        self.upconv_1 = nn.ConvTranspose3d(self.event_num, 256, kernel_size = 3, stride = 2, padding = 0, output_padding = 0)
        self.upconv_2 = nn.ConvTranspose3d(256, 32, kernel_size = 3, stride = 2,  padding = 0, output_padding = 0)
        self.upconv_3 = nn.ConvTranspose3d(32, 16, kernel_size = 3, stride = 1,  padding = 0, output_padding = 0)
        self.upconv_4 = nn.ConvTranspose3d(16, 8, kernel_size = 3, stride = 1,  padding = 0, output_padding = 0)
        self.upconv_5 = nn.ConvTranspose3d(8, 1, kernel_size = 3, stride = 1,  padding = 0, output_padding = 0)


    def forward(self, x):
        bs = x.shape[0] # batch size
        x = x.view(bs, -1, self.small_cube_side, self.small_cube_side, self.small_cube_side) # small cubes
        x = self.relu(self.upconv_1(x))
        x = self.relu(self.upconv_2(x))
        x = self.relu(self.upconv_3(x))
        x = self.relu(self.upconv_4(x))
        x = self.upconv_5(x)
        x = x.squeeze(1)
        return x
    
class EventsToPerm(nn.Module):
    '''
    Model itself
    list of events -> permeability map
    '''
    def __init__(self, params):
        super().__init__()
        self.input_shape = (1100, 5)
        self.target_shape = (21, 21, 21)
        self.inp_size = self.input_shape[1] # 5
        self.event_num = self.input_shape[0] # 1100
        self.small_cube_side = params.small_cube_side
        self.emb_size = self.small_cube_side ** 3

        self.embed = Fc(inp_size = self.inp_size, target_size = self.emb_size, num_layers = params.embed_num_layers, flat=params.flat, fc_drop = 0.0)
        self.encoder = EventListTransformerEncoder(emb_size = self.emb_size, num_layers = params.encoder_num_layers, nheads = params.nheads)
        self.upconv = Upconv(event_num = self.event_num,  small_cube_side = self.small_cube_side, target_shape = self.target_shape)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = self.upconv(x)
        return x

# Net class with training evaluating and other---------------------------------------------------------- 

class PermNet:
    def __init__(self, train_h5_path, test_h5_path, train_params):
        self.params = train_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dl, self.test_dl = self.create_dl(train_h5_path, test_h5_path)
        self.model = EventsToPerm(self.params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.mask = self.get_ev_dens_mask(train_h5_path)
        self.loss = MaskedMSELoss(self.mask)

    def __call__(self, X):
        self.model.to(self.device).eval()
        X = X.to(self.device)
        with torch.no_grad():
            y_hat = self.model(X).cpu()
        self.model.to("cpu")
        return y_hat

    def evaluate(self):
        val_loss = []
        self.model.to(self.device).eval()
        for X, y in self.test_dl:
            X, y = X.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(X)
            l = self.loss(y_hat, y)
            val_loss.append(l.item())
        self.model.to("cpu")
        return torch.mean(torch.tensor(val_loss)).item()

    def train(self):
        self.metrics = {'loss':{'train': [], 'val': []}, 'epoch': []}
        for epoch in tqdm(range(self.params.epochs)):
            losses = self.train_one_epoch()
            self.metrics['epoch'].append(epoch)
            self.metrics['loss']['train'].append(losses['train'])
            self.metrics['loss']['val'].append(losses['val'])
            if epoch % 1 == 0:
                print(f"Epoch {epoch}:: train loss: {self.metrics['loss']['train'][-1]:.04f}, val loss: {self.metrics['loss']['val'][-1]:.04f}")
        return self.metrics

    def train_one_epoch(self):
        self.model.to(self.device).train()
        for X, y in tqdm(self.train_dl):
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.model(X)
            l = self.loss(y_hat, y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
        losses = {'train': l.item(), 'val': self.evaluate()}
        self.model.to("cpu")
        return losses

    def create_dl(self, train_h5_path, test_h5_path):
        # creates dataloaders
        transforms = [ToTensor(), Normalizer(),] # data transformations

        train = EventsToPermDataset(h5_path = train_h5_path, transforms=transforms)
        test = EventsToPermDataset(h5_path = test_h5_path, transforms=transforms)

        train_dl = DataLoader(dataset=train,
                            batch_size=self.params.batch_size,
                            shuffle=True,
                            )

        test_dl = DataLoader(dataset=test,
                            batch_size=self.params.batch_size,
                            shuffle=False,
                            )

        return train_dl, test_dl

    def get_ev_dens_mask(self, train_h5_path, eps=1e-3):
        with h5py.File(train_h5_path, 'a') as f:
            ev_d = f['ev_dens'][:].astype('float32')
            mean_ev_d = np.mean(ev_d, axis=0)
            mask = mean_ev_d/np.max(mean_ev_d)
            mask[mask<eps] = eps
        
        return torch.tensor(mask).to(self.device)

    def plot_metrics(self):
        fig, ax = plt.subplots(figsize=(4.33, 3))
        ax.plot(self.metrics['epoch'], self.metrics['loss']['train'], label='train')
        ax.plot(self.metrics['epoch'], self.metrics['loss']['val'],  label='val')
        ax.set_title('Loss', loc='center', fontsize=12)
        ax.set_xlabel('# Epoch', fontsize=12)
        ax.legend()
        
    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def load_weights(self, pt_path):
        self.model.load_state_dict(torch.load(pt_path)) 

    def save(self, state_name=''):
        Path(f'{CURR_DIR}/model_states/').mkdir(parents=True, exist_ok=True)
        pt_path = f'{CURR_DIR}/model_states/{state_name}{datetime.now().strftime("%m_%d_%Y__%H_%M_%S")}.pt'
        torch.save(self.model.state_dict(), pt_path)

# train config class ------------------------------------------------------------

class TrainConfig:
    def __init__(self, **kwargs):
        # embed layer params
        self.embed_num_layers = 5
        self.flat = True

        # small cube
        self.small_cube_side = 3
        
        # encoder params
        self.encoder_num_layers = 2
        self.nheads = 1
        
        # training params
        self.batch_size = 60
        self.epochs = 35
        self.lr = 0.0001

        # # device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # kwargs to attrs
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:
        return str(self.__dict__)