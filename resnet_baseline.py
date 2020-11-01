"""
Thank you to the following notebooks:
https://www.kaggle.com/lucabergamini/lyft-baseline-09-02/
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
        'fname': "baseline_models/baseline_best.pth"
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

def build_model(cfg:Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet50(pretrained=True)

    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

def forward(data, model, device, criterion = nn.MSELoss(reduction="none")):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = (criterion(outputs, targets) * target_availabilities).mean()
    
    return outputs, loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)
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