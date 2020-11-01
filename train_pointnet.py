from abc import ABC
from pathlib import Path
from numcodecs import blosc
import pandas as pd, numpy as np
import os

import bisect
import itertools as it
from tqdm import tqdm
import logzero

import torch
from torch import nn, optim 
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger

import pickle, copy, re, time, datetime, random, warnings, gc
import zarr

from poinet_model import *

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
EPOCHS = JSON_PARAMETERS["EPOCHS"]
LEARNING_RATE = JSON_PARAMETERS["LEARNING_RATE"]
WEIGHT_DECAY = JSON_PARAMETERS["WEIGHT_DECAY"]
GRADIENT_CLIP_VAL = JSON_PARAMETERS["GRADIENT_CLIP_VAL"]
LIMIT_VAL_BATCHES = JSON_PARAMETERS["LIMIT_VAL_BATCHES"]

# KEPT_PERCEPTION_LABELS_DICT = {label:PERCEPTION_LABELS.index(label) for label in KEPT_PERCEPTION_LABELS}
# KEPT_PERCEPTION_KEYS = sorted(KEPT_PERCEPTION_LABELS_DICT.values())

torch.backends.cudnn.benchmark =  True

# last_checkpoint = get_last_checkpoint(ROOT)
last_checkpoint = None

if last_checkpoint is not None:
    print(f'\n***** RESUMING FROM CHECKPOINT `{last_checkpoint.as_posix()}`***********\n')
    model = LyftNet.load_from_checkpoint(Path(last_checkpoint).as_posix(), 
    map_location=device, num_workers = NUM_WORKERS, batch_size = BATCH_SIZE)
else:
    print('\n***** NEW MODEL ***********\n')
    model = LyftNet(batch_size=BATCH_SIZE, 
                lr= LEARNING_RATE, weight_decay=WEIGHT_DECAY, num_workers=NUM_WORKERS)

checkpoint_callback = ModelCheckpoint(
    filepath=ROOT,
    save_top_k=5,
    verbose=0,
    monitor='val_loss',
    mode='min',
    prefix='lyfnet_',
)

API_KEY = os.environ.get('NEPTUNE_API_KEY')
neptune_logger = NeptuneLogger(
                api_key=API_KEY,
                project_name='hvergnes/KagglePointNet',
                params={'epoch_nr': f'{EPOCHS}', 'bs': f'{BATCH_SIZE}', 'LEARNING_RATE': f'{LEARNING_RATE}', 'WEIGHT_DECAY': f'{WEIGHT_DECAY}',  'HBACKWARD': f'{HBACKWARD}',
                'HFORWARD': f'{HFORWARD}', 'NFRAMES': f'{NFRAMES}', "FRAME_STRIDE": f"{FRAME_STRIDE}", "AGENT_FEATURE_DIM": f"{AGENT_FEATURE_DIM}",
                "MAX_AGENTS": f"{MAX_AGENTS}"},
                tags=['baseline'],
                )

# print(model)
trainer = Trainer(
    max_epochs=EPOCHS,
    gradient_clip_val=GRADIENT_CLIP_VAL,
    logger=neptune_logger,
    checkpoint_callback=checkpoint_callback,
    limit_val_batches=LIMIT_VAL_BATCHES,
    gpus=1
)

trainer.fit(model)
torch.save(model.state_dict(), f'save/PointNetE:{EPOCHS}LR:{LEARNING_RATE}.pt')
