"""
Thank you to the following notebooks:
https://www.kaggle.com/lucabergamini/lyft-baseline-09-02/
https://www.kaggle.com/kramadhari/lstm-based-encoder-decoder-seq2seq-for-mp
https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence
"""

# ensure version of L5Kit
import l5kit
assert l5kit.__version__ == "1.1.0"

import numpy as np
import pandas as pd
import os
import torch
import time

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm.notebook import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.geometry import transform_points
from l5kit.dataset import AgentDataset
from l5kit.evaluation import write_pred_csv
from l5kit.rasterization import build_rasterizer

os.environ["L5KIT_DATA_FOLDER"] = "/data/"
dm = LocalDataManager(None)

cfg = {
    'format_version': 4,
    'model_params': {
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'lr': 1e-3,
        'fname': "lstm_model/lstm_best.pth"
    },
    
    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    },
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    },
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4
    },
    'train_params': {
        'max_num_steps': 10000,
        'checkpoint_every_n_steps': 2000,
    }

}

train_cfg = cfg["train_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

rasterizer = build_rasterizer(cfg, dm)
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"]
                             )
print(train_dataset)

val_cfg = cfg["val_data_loader"]
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()

rasterizer = build_rasterizer(cfg, dm)
val_dataset = AgentDataset(cfg, val_zarr, rasterizer)
val_dataloader = DataLoader(val_dataset,
                              shuffle=val_cfg["shuffle"],
                              batch_size=val_cfg["batch_size"],
                              num_workers=val_cfg["num_workers"]
                             )
print(val_dataset)

class EncoderLSTM_LyftModel(nn.Module):
    
    def __init__(self, cfg):
        super(EncoderLSTM_LyftModel, self).__init__()
        
        self.input_sz  = 2
        self.hidden_sz = 128
        self.num_layer = 1
        self.sequence_length = 11        
        
        self.Encoder_lstm = nn.LSTM(self.input_sz,self.hidden_sz,self.num_layer,batch_first=True)
       
    def forward(self,inputs):
        output,hidden_state = self.Encoder_lstm(inputs)
        return output,hidden_state
    
class DecoderLSTM_LyftModel(nn.Module):
    def __init__(self, cfg):
        super(DecoderLSTM_LyftModel, self).__init__()
        
        self.input_sz  = 128 #(2000 from fcn_en_output reshape to 50*40)
        self.hidden_sz = 128
        self.hidden_sz_en = 128
        self.num_layer = 1
        self.sequence_len_de = 1
        
        self.interlayer = 256

        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        
        self.encoderLSTM = EncoderLSTM_LyftModel (cfg)

        
        self.Decoder_lstm = nn.LSTM( self.input_sz,self.hidden_sz,self.num_layer,batch_first=True)

        
        self.fcn_en_state_dec_state= nn.Sequential(nn.Linear(in_features=self.hidden_sz_en, out_features=self.interlayer),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_features=self.interlayer, out_features=num_targets))

    def forward(self,inputs):
        
        _,hidden_state = self.encoderLSTM(inputs)
        
        inout_to_dec = torch.ones(inputs.shape[0],self.sequence_len_de,self.input_sz).to(device)

        #for i in range(cfg["model_params"]["future_num_frames"]+1): # this can be used to feed output from previous LSTM to anther one which is stacked.
        inout_to_dec,hidden_state   = self.Decoder_lstm(inout_to_dec,(hidden_state[0],hidden_state[1]) )          
        
        fc_out = self.fcn_en_state_dec_state (inout_to_dec.squeeze(dim=0))
        
        return fc_out.reshape(inputs.shape[0],cfg["model_params"]["future_num_frames"],-1)

def forward(data, model, device, criterion = nn.MSELoss(reduction="none")):
    history_positions = data['history_positions'].to(device)
    history_availabilities = data['history_availabilities'].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets_position = data["target_positions"].to(device)

    outputs = model(history_positions)
    loss = (criterion(outputs, targets_position) * target_availabilities).mean()
    
    return outputs, loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DecoderLSTM_LyftModel(cfg).to(device)
if os.path.exists(cfg["model_params"]["fname"]):
    model.load_state_dict(torch.load(cfg["model_params"]["fname"], map_location=device))

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
print(f'device {device}')


tr_it = iter(train_dataloader)
vl_it = iter(val_dataloader)

progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

losses_train = []
losses_val = []

iterations = []
metrics = []
times = []

start = time.time()
for i in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)
    
    # Forward pass
    _, loss = forward(data, model, device)
    losses_train.append(loss.item())
    
    # Get validation loss before backward pass
    with torch.no_grad():
        try:
            val_data = next(vl_it)
        except StopIteration:
            vl_it = iter(val_dataloader)
            val_data = next(vl_it)

        model.eval()
        outputs_val, loss_val = forward(val_data, model, device)
        losses_val.append(loss_val.item())
        if loss_val == min(losses_val):
            torch.save(model.state_dict(), f'{cfg["model_params"]["fname"]}')
            
    desc = f" TrainLoss: {round(loss.item(), 4)} ValLoss: {round(loss_val.item(), 4)} TrainMeanLoss: {round(np.mean(losses_train),4)} ValMeanLoss: {round(np.mean(losses_val),4)}" 
    print(desc)
    # progress_bar.set_description(desc)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Total training time is {(time.time()-start)/60} mins")


vl_it = iter(val_dataloader)
progress_bar = tqdm(val_dataloader)
losses_val = []
with torch.no_grad():
    for i in progress_bar:
        try:
            val_data = next(vl_it)
        except StopIteration:
            vl_it = iter(val_dataloader)
            val_data = next(vl_it)

        model.eval()
        outputs_val, loss_val = forward(val_data, model, device)
        losses_val.append(loss_val.item())

print("Validation loss: {}".format(np.mean(losses_val),4))
