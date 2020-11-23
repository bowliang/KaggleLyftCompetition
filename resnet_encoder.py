

import l5kit
assert l5kit.__version__ == "1.1.0"

import numpy as np
import pandas as pd
import os
import torch
import time

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet18
from tqdm.notebook import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.geometry import transform_points
from l5kit.dataset import AgentDataset
from l5kit.evaluation import write_pred_csv
from l5kit.rasterization import build_rasterizer


mybatchsize = 16


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
        'out_path': "resnet_decoder"
    },
    
    'raster_params': {
        'raster_size': [256, 256],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': mybatchsize,
        'shuffle': True,
        'num_workers': 4
    },
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': mybatchsize,
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
out_path = cfg["model_params"]["out_path"]


# In[ ]:


train_cfg = cfg["train_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

rasterizer = build_rasterizer(cfg, dm)
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"]
                             )
# print(train_dataset)
print("loaded training dataset")

val_cfg = cfg["val_data_loader"]
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()

rasterizer = build_rasterizer(cfg, dm)
val_dataset = AgentDataset(cfg, val_zarr, rasterizer)
val_dataloader = DataLoader(val_dataset,
                              shuffle=val_cfg["shuffle"],
                              batch_size=val_cfg["batch_size"],
                              num_workers=val_cfg["num_workers"]
                             )
# print(val_dataset)
print("loaded validation dataset")


# In[ ]:


import torch
from torch import Tensor

def pytorch_neg_multi_log_likelihood_batch(gt, pred, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return 3*torch.mean(error)


# In[ ]:


class DecoderLSTM_LyftModel(nn.Module):
    def __init__(self, cfg):
        super(DecoderLSTM_LyftModel, self).__init__()
        
        self.resnet = resnet18(pretrained=True)
        num_in_channels = 2
        self.resnet.conv1 = nn.Conv2d(
            num_in_channels,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=False,
        )
        self.resnet.fc = nn.Linear(in_features=512, out_features=512)
        
        self.input_sz  = 512
        self.hidden_sz = 512
        self.hidden_sz_en = 512
        self.num_layer = 1
        self.sequence_len_de = 1
        self.num_modes = 3
        
        self.interlayer = 512
        
        self.future_len = cfg["model_params"]["future_num_frames"]
        self.num_targets = 2 * self.future_len
        self.num_preds = self.num_targets * self.num_modes
        
        self.Decoder_lstm = nn.LSTM(self.input_sz, self.hidden_sz, self.num_layer, batch_first=True)

        
        self.fcn_en_state_dec_state= nn.Sequential(nn.Linear(in_features=self.hidden_sz_en, out_features=self.interlayer),
                            nn.Dropout(0.5, inplace=True),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_features=self.interlayer, out_features=self.num_preds + self.num_modes))

    def forward(self,inputs):
        bsnf, _, _, _ = inputs.shape
        assert bsnf == mybatchsize * 11
        
        x = self.resnet(inputs)
        # assert tuple(x.shape) == (88, 1, 128)
        
        x = x.view(mybatchsize, 11, 512)
        inout_to_dec, hidden_state = self.Decoder_lstm(x) 
        # assert tuple(inout_to_dec.shape) == (8, 11, 128)
        
        inout_to_dec = inout_to_dec[:, -1, :].view(mybatchsize, 1, 512)
        # assert tuple(inout_to_dec.shape) == (8, 1, 128)
        
        x = self.fcn_en_state_dec_state (inout_to_dec.squeeze(dim=0))
        # assert x.shape == (8, 1, 128)

        bs = x.shape[0]
        pred, confidences = x[:, :, :-3], x[:, :, -3:]
        
        confidences = confidences.squeeze()
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        confidences = confidences.view(bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    
def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    x = data['image']
    batchsize, numframes, _, _ = x.shape
    
    x = x[:, :-3, :, :]
    agents, ego = torch.split(x, 11, dim =1)
    # print(agents.shape)
    # print(ego.shape)
    
    agents = torch.reshape(agents, (mybatchsize * 11, 1, 256, 256))
    ego = torch.reshape(ego, (mybatchsize * 11, 1, 256, 256))
    
    x = torch.cat([agents, ego], 1).to(device)
    
    
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    
    # Forward pass
    preds, confidences = model(x)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DecoderLSTM_LyftModel(cfg).to(device)

try:
    if os.path.exists(f"{out_path}/best_val.pth"): model.load_state_dict(torch.load(f"{out_path}/best_val.pth", map_location=device))
except:
    pass

if not os.path.exists(out_path): os.makedirs(out_path)
        
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
print(f'device {device}')


# In[ ]:


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
    loss, _, _ = forward(data, model, device)
    losses_train.append(loss.item())
    
    # Get validation loss before backward pass
    if i % 50 == 0:
        with torch.no_grad():
            try:
                val_data = next(vl_it)
            except StopIteration:
                vl_it = iter(val_dataloader)
                val_data = next(vl_it)

            model.eval()
            loss_val, _, _ = forward(val_data, model, device)
            losses_val.append(loss_val.item())
            if loss_val == min(losses_val):
                torch.save(model.state_dict(), f"{out_path}/best_val.pth")
                os.system(f"echo {loss_val.item()} >> {out_path}/log")
                print(f"Saved model with ValLoss: {round(loss_val.item(), 4)}")
        print(f" TrainLoss: {round(loss.item(), 4)} ValLoss: {round(loss_val.item(), 4)} TrainMeanLoss: {round(np.mean(losses_train),4)} ValMeanLoss: {round(np.mean(losses_val),4)}")
            
    desc = f" TrainLoss: {round(loss.item(), 4)} TrainMeanLoss: {round(np.mean(losses_train),4)}" 
    print(desc)
    progress_bar.set_description(desc)

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
        loss_val, _, _ = forward(val_data, model, device)
        losses_val.append(loss_val.item())

print("Validation loss: {}".format(np.mean(losses_val),4))


# In[ ]:





# In[ ]:





