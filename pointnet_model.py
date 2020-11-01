# Thanks to KKiller on Kaggle for designing this model.
from torch.utils.data import Dataset, DataLoader
from abc import ABC
from pathlib import Path
from numcodecs import blosc
import pandas as pd, numpy as np

import bisect
import itertools as it
from tqdm import tqdm
import logzero
import json


import torch
from torch import nn, optim 
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import pickle, copy, re, time, datetime, random, warnings, gc
import zarr

with open('parameters.json') as json_file:
    JSON_PARAMETERS = json.load(json_file)

DATA_ROOT = Path("/data/lyft-motion-prediction-autonomous-vehicles")
TRAIN_ZARR = JSON_PARAMETERS["TRAIN_ZARR"]
VALID_ZARR = JSON_PARAMETERS["VALID_ZARR"]

HBACKWARD = JSON_PARAMETERS["HBACKWARD"]
HFORWARD = JSON_PARAMETERS["HFORWARD"]
NFRAMES = JSON_PARAMETERS["NFRAMES"]
FRAME_STRIDE = JSON_PARAMETERS["FRAME_STRIDE"]
AGENT_FEATURE_DIM = JSON_PARAMETERS["AGENT_FEATURE_DIM"]
MAX_AGENTS = JSON_PARAMETERS["MAX_AGENTS"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = JSON_PARAMETERS["NUM_WORKERS"]
BATCH_SIZE = JSON_PARAMETERS["BATCH_SIZE"]
EPOCHS=JSON_PARAMETERS["EPOCHS"]
GRADIENT_CLIP_VAL = JSON_PARAMETERS["GRADIENT_CLIP_VAL"]
LIMIT_VAL_BATCHES = JSON_PARAMETERS["LIMIT_VAL_BATCHES"]
ROOT = JSON_PARAMETERS["ROOT"]

Path(ROOT).mkdir(exist_ok=True, parents=True)

def get_utc():
    TIME_FORMAT = r"%Y-%m-%dT%H:%M:%S%Z"
    return datetime.datetime.now(datetime.timezone.utc).strftime(TIME_FORMAT)

PERCEPTION_LABELS = JSON_PARAMETERS["PERCEPTION_LABELS"]
KEPT_PERCEPTION_LABELS = JSON_PARAMETERS["KEPT_PERCEPTION_LABELS"]

KEPT_PERCEPTION_LABELS_DICT = {label:PERCEPTION_LABELS.index(label) for label in KEPT_PERCEPTION_LABELS}
KEPT_PERCEPTION_KEYS = sorted(KEPT_PERCEPTION_LABELS_DICT.values()) 
class LabelEncoder:
    def  __init__(self, max_size=500, default_val=-1):
        self.max_size = max_size
        self.labels = {}
        self.default_val = default_val

    @property
    def nlabels(self):
        return len(self.labels)

    def reset(self):
        self.labels = {}

    def partial_fit(self, keys):
        nlabels = self.nlabels
        available = self.max_size - nlabels

        if available < 1:
            return

        keys = set(keys)
        new_keys = list(keys - set(self.labels))

        if not len(new_keys):
            return
        
        self.labels.update(dict(zip(new_keys, range(nlabels, nlabels + available) )))
    
    def fit(self, keys):
        self.reset()
        self.partial_fit(keys)

    def get(self, key):
        return self.labels.get(key, self.default_val)
    
    def transform(self, keys):
        return np.array(list(map(self.get, keys)))

    def fit_transform(self, keys, partial=True):
        self.partial_fit(keys) if partial else self.fit(keys)
        return self.transform(keys)

class CustomLyftDataset(Dataset):
    feature_mins = np.array([-17.336, -27.137, 0. , 0., 0. , -3.142, -37.833, -65.583],
    dtype="float32")[None,None, None]

    feature_maxs = np.array([17.114, 20.787, 42.854, 42.138,  7.079,  3.142, 29.802, 35.722],
    dtype="float32")[None,None, None]



    def __init__(self, zdataset, scenes=None, nframes=10, frame_stride=15, hbackward=10, 
                 hforward=50, max_agents=150, agent_feature_dim=8):
        """
        Custom Lyft dataset reader.
        
        Parmeters:
        ----------
        zdataset: zarr dataset
            The root dataset, containing scenes, frames and agents
            
        nframes: int
            Number of frames per scene
            
        frame_stride: int
            The stride when reading the **nframes** frames from a scene
            
        hbackward: int
            Number of backward frames from  current frame
            
        hforward: int
            Number forward frames from current frame
        
        max_agents: int 
            Max number of agents to read for each target frame. Note that,
            this also include the backward agents but not the forward ones.
        """
        super().__init__()
        self.zdataset = zdataset
        self.scenes = scenes if scenes is not None else []
        self.nframes = nframes
        self.frame_stride = frame_stride
        self.hbackward = hbackward
        self.hforward = hforward
        self.max_agents = max_agents

        self.nread_frames = (nframes-1)*frame_stride + hbackward + hforward

        self.frame_fields = ['timestamp', 'agent_index_interval']

        self.agent_feature_dim = agent_feature_dim

        self.filter_scenes()
      
    def __len__(self):
        return len(self.scenes)

    def filter_scenes(self):
        self.scenes = [scene for scene in self.scenes if self.get_nframes(scene) > self.nread_frames]


    def __getitem__(self, index):
        return self.read_frames(scene=self.scenes[index])

    def get_nframes(self, scene, start=None):
        frame_start = scene["frame_index_interval"][0]
        frame_end = scene["frame_index_interval"][1]
        nframes = (frame_end - frame_start) if start is None else ( frame_end - max(frame_start, start) )
        return nframes


    def _read_frames(self, scene, start=None):
        nframes = self.get_nframes(scene, start=start)
        assert nframes >= self.nread_frames

        frame_start = scene["frame_index_interval"][0]

        start = start or frame_start + np.random.choice(nframes-self.nread_frames)
        frames = self.zdataset.frames.get_basic_selection(
            selection=slice(start, start+self.nread_frames),
            fields=self.frame_fields,
            )
        return frames
    

    def parse_frame(self, frame):
        return frame

    def parse_agent(self, agent):
        return agent

    def read_frames(self, scene, start=None,  white_tracks=None, encoder=False):
        white_tracks = white_tracks or []
        frames = self._read_frames(scene=scene, start=start)

        agent_start = frames[0]["agent_index_interval"][0]
        agent_end = frames[-1]["agent_index_interval"][1]

        agents = self.zdataset.agents[agent_start:agent_end]


        X = np.zeros((self.nframes, self.max_agents, self.hbackward, self.agent_feature_dim), dtype=np.float32)
        target = np.zeros((self.nframes, self.max_agents, self.hforward, 2),  dtype=np.float32)
        target_availability = np.zeros((self.nframes, self.max_agents, self.hforward), dtype=np.uint8)
        X_availability = np.zeros((self.nframes, self.max_agents, self.hbackward), dtype=np.uint8)

        for f in range(self.nframes):
            backward_frame_start = f*self.frame_stride
            forward_frame_start = f*self.frame_stride+self.hbackward
            backward_frames = frames[backward_frame_start:backward_frame_start+self.hbackward]
            forward_frames = frames[forward_frame_start:forward_frame_start+self.hforward]

            backward_agent_start = backward_frames[-1]["agent_index_interval"][0] - agent_start
            backward_agent_end = backward_frames[-1]["agent_index_interval"][1] - agent_start

            backward_agents = agents[backward_agent_start:backward_agent_end]

            le = LabelEncoder(max_size=self.max_agents)
            le.fit(white_tracks)
            le.partial_fit(backward_agents["track_id"])

            for iframe, frame in enumerate(backward_frames):
                backward_agent_start = frame["agent_index_interval"][0] - agent_start
                backward_agent_end = frame["agent_index_interval"][1] - agent_start

                backward_agents = agents[backward_agent_start:backward_agent_end]

                track_ids = le.transform(backward_agents["track_id"])
                mask = (track_ids != le.default_val)
                mask_agents = backward_agents[mask]
                mask_ids = track_ids[mask]
                X[f, mask_ids, iframe, :2] = mask_agents["centroid"]
                X[f, mask_ids, iframe, 2:5] = mask_agents["extent"]
                X[f, mask_ids, iframe, 5] = mask_agents["yaw"]
                X[f, mask_ids, iframe, 6:8] = mask_agents["velocity"]

                X_availability[f, mask_ids, iframe] = 1

            
            for iframe, frame in enumerate(forward_frames):
                forward_agent_start = frame["agent_index_interval"][0] - agent_start
                forward_agent_end = frame["agent_index_interval"][1] - agent_start

                forward_agents = agents[forward_agent_start:forward_agent_end]

                track_ids = le.transform(forward_agents["track_id"])
                mask = track_ids != le.default_val

                target[f, track_ids[mask], iframe] = forward_agents[mask]["centroid"]
                target_availability[f, track_ids[mask], iframe] = 1

        target -= X[:,:,[-1], :2]
        target *= target_availability[:,:,:,None]
        X[:,:,:, :2] -= X[:,:,[-1], :2]
        X *= X_availability[:,:,:,None]
        X -= self.feature_mins
        X /= (self.feature_maxs - self.feature_mins)

        if encoder:
            return X, target, target_availability, le
        return X, target, target_availability

def collate(x):
    x = map(np.concatenate, zip(*x))
    x = map(torch.from_numpy, x)
    return x

def shapefy( xy_pred, xy, xy_av):
    NDIM = 3
    xy_pred = xy_pred.view(-1, HFORWARD, NDIM, 2)
    xy = xy.view(-1, HFORWARD, 2)[:,:,None]
    xy_av = xy_av.view(-1, HFORWARD)[:,:,None]
    return xy_pred, xy, xy_av


def LyftLoss(c, xy_pred, xy, xy_av):
    c = c.view(-1, c.shape[-1])
    xy_pred, xy, xy_av  = shapefy(xy_pred, xy, xy_av)
    
    c = torch.softmax(c, dim=1)
    
    l = torch.sum(torch.mean(torch.square(xy_pred-xy), dim=3)*xy_av, dim=1)
    
    # The LogSumExp trick for better numerical stability
    # https://en.wikipedia.org/wiki/LogSumExp
    m = l.min(dim=1).values
    l = torch.exp(m[:, None]-l)
    
    l = m - torch.log(torch.sum(l*c, dim=1))
    denom = xy_av.max(2).values.max(1).values
    l = torch.sum(l*denom)/denom.sum()
    return 3*l # I found that my loss is usually 3 times smaller than the LB score

def MSE(xy_pred, xy, xy_av):
    xy_pred, xy, xy_av = shapefy(xy_pred, xy, xy_av)
    return 9*torch.mean(torch.sum(torch.mean(torch.square(xy_pred-xy), 3)*xy_av, dim=1))

def MAE(xy_pred, xy, xy_av):
    xy_pred, xy, xy_av = shapefy(xy_pred, xy, xy_av)
    return 9*torch.mean(torch.sum(torch.mean(torch.abs(xy_pred-xy), 3)*xy_av, dim=1))

class BaseNet(LightningModule):   
    def __init__(self, batch_size=32, lr=5e-4, weight_decay=1e-8, num_workers=0, 
                 criterion=LyftLoss, data_root=DATA_ROOT,  epochs=1):
        super().__init__()

       
        self.save_hyperparameters(
            dict(
                HBACKWARD = HBACKWARD,
                HFORWARD = HFORWARD,
                NFRAMES = NFRAMES,
                FRAME_STRIDE = FRAME_STRIDE,
                AGENT_FEATURE_DIM = AGENT_FEATURE_DIM,
                MAX_AGENTS = MAX_AGENTS,
                TRAIN_ZARR = TRAIN_ZARR,
                VALID_ZARR = VALID_ZARR,
                batch_size = batch_size,
                lr=lr,
                weight_decay=weight_decay,
                num_workers=num_workers,
                criterion=criterion,
                epochs=epochs,
            )
        )
        
        self._train_data = None
        self._collate_fn = None
        self._train_loader = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
        self.lr = lr
        self.epochs=epochs
        
        self.weight_decay = weight_decay
        self.criterion = criterion
        
        self.data_root = data_root
    

    def train_dataloader(self):
        z = zarr.open(self.data_root.joinpath(TRAIN_ZARR).as_posix(), "r")
        scenes = z.scenes.get_basic_selection(slice(None), fields= ["frame_index_interval"])
        train_data = CustomLyftDataset(
                    z, 
                    scenes = scenes,
                    nframes=NFRAMES,
                    frame_stride=FRAME_STRIDE,
                    hbackward=HBACKWARD,
                    hforward=HFORWARD,
                    max_agents=MAX_AGENTS,
                    agent_feature_dim=AGENT_FEATURE_DIM,
                )
        
        train_loader = DataLoader(train_data, batch_size = self.batch_size,collate_fn=collate,
                                pin_memory=True, num_workers = self.num_workers, shuffle=True)
        self._train_data = train_data
        self._train_loader = train_loader
        
        return train_loader

    def val_dataloader(self):
        z = zarr.open(self.data_root.joinpath(VALID_ZARR).as_posix(), "r")
        scenes = z.scenes.get_basic_selection(slice(None), fields=["frame_index_interval"])
        val_data = CustomLyftDataset(
                    z, 
                    scenes = scenes,
                    nframes=NFRAMES,
                    frame_stride=FRAME_STRIDE,
                    hbackward=HBACKWARD,
                    hforward=HFORWARD,
                    max_agents=MAX_AGENTS,
                    agent_feature_dim=AGENT_FEATURE_DIM,
                )
        
        val_loader = DataLoader(val_data, batch_size = self.batch_size, collate_fn=collate,
                                pin_memory=True, num_workers = self.num_workers, shuffle=True)
        self._val_data = val_data
        self._val_loader = val_loader
        return val_loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.tensor([x['val_loss'] for x in outputs]))
        avg_mse = torch.mean(torch.tensor([x['val_mse'] for x in outputs]))
        avg_mae = torch.mean(torch.tensor([x['val_mae'] for x in outputs]))
        
        tensorboard_logs = {'val_loss': avg_loss, "val_rmse": torch.sqrt(avg_mse), "val_mae": avg_mae}

        torch.cuda.empty_cache()
        gc.collect()

        return {
            'val_loss': avg_loss,
            'log': tensorboard_logs,
            "progress_bar": {"val_ll": tensorboard_logs["val_loss"], "val_rmse": tensorboard_logs["val_rmse"]}
        }

    
    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr= self.lr, betas= (0.9,0.999), 
                          weight_decay= self.weight_decay, amsgrad=False)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-5,
        )
        return [optimizer], [scheduler]

class STNkd(nn.Module):
    def __init__(self,  k=64):
        super(STNkd, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(k, 256, kernel_size=1), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=1), nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=1), nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, k*k),nn.ReLU(),
        )
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv(x)
        x = torch.max(x, 2)[0]
        x = self.fc(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,
                                                                            self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = False, feature_transform = False, stn1_dim = 120,
                 stn2_dim = 64):
        super(PointNetfeat, self).__init__()
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.stn1_dim = stn1_dim
        self.stn2_dim = stn2_dim
        
        self.stn = STNkd(k=stn1_dim)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(stn1_dim, 256, kernel_size=1), nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1), nn.ReLU(),
            nn.Conv1d(256, 1024, kernel_size=1), nn.ReLU(),
            nn.Conv1d(1024, 2048, kernel_size=1), nn.ReLU(),
        )
        
        if self.feature_transform:
            self.fstn = STNkd(k=stn2_dim)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        x = self.conv1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        
        x = self.conv2(x)
        x = torch.max(x, 2)[0]
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x[:,:,None].repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class LyftNet(BaseNet):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.pnet = PointNetfeat()

        self.fc0 = nn.Sequential(
            nn.Linear(2048+256, 1024), nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 300),
        )

        self.c_net = nn.Sequential(
            nn.Linear(1024, 3),
        )
        
    
    def forward(self, x):
        bsize, npoints, hb, nf = x.shape 
        
        # Push points to the last  dim
        x = x.transpose(1, 3)

        # Merge time with features
        x = x.reshape(bsize, hb*nf, npoints)

        x, trans, trans_feat = self.pnet(x)

        # Push featuresxtime to the last dim
        x = x.transpose(1,2)

        x = self.fc0(x)

        c = self.c_net(x)
        x = self.fc(x)

        return c,x
    
    def training_step(self, batch, batch_idx):
        x, y, y_av = [b.to(device) for b in batch]
        c, preds = self(x)
        loss = self.criterion(c,preds,y, y_av)
        
        with torch.no_grad():
            logs = {
                'loss': loss,
                "mse": MSE(preds, y, y_av),
                "mae": MAE(preds, y, y_av),
            }
        return {'loss': loss, 'log': logs, "progress_bar": {"rmse":torch.sqrt(logs["mse"]) }}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y, y_av =  [b.to(device) for b in batch]
        c,preds = self(x)
        loss = self.criterion(c, preds, y, y_av)
        
        val_logs = {
            'val_loss': loss,
            "val_mse": MSE(preds, y, y_av),
            "val_mae": MAE(preds, y, y_av),
        }
        
        return val_logs

def get_last_checkpoint(root):
    res = None
    mtime = -1
    for model_name in Path(root).glob("lyfnet*.ckpt"):
        e = model_name.stat().st_ctime
        if e > mtime:
            mtime=e
            res = model_name
    return res

def get_last_version(root):

    last_version = 0
    for model_name in Path(root).glob("version_*"):
        version = int(model_name.as_posix().split("_")[-1])
        if version > last_version:
            last_version = version
    return last_version
