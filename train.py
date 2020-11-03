# l5kit
import l5kit, os, albumentations as A
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, PERCEPTION_LABELS
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.evaluation.csv_utils import write_pred_csv

from catalyst import dl
from catalyst.dl import utils
from collections import Counter
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from tqdm import tqdm


from model import LyftMultiModel, set_seed
# from logzero import logger

set_seed(42)

# Hyperparameters
cfg = load_config_data("/data/lyft-motion-prediction-autonomous-vehicles/lyft-config-files/agent_motion_config.yaml")
cfg = omegaconf.DictConfig(cfg)
name_for_save = 'Big_training'
epochs = cfg["model_params"]["epochs"]
learning_rate = cfg["model_params"]["lr"]
training_percentage = 0.1
validation_percentage = 1

API_KEY = os.environ.get('NEPTUNE_API_KEY')
neptune_logger = NeptuneLogger(
                api_key=API_KEY,
                project_name='hvergnes/KaggleResNet',
                params={'epoch_nr': epochs, 'learning_rate': learning_rate, 'train_size': training_percentage, 'test_size': validation_percentage},  # your hyperparameters, immutable
                tags=['ResNet'],  # tags
                )


os.environ["L5KIT_DATA_FOLDER"] = "/data/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager()

cfg = load_config_data("/data/lyft-motion-prediction-autonomous-vehicles/lyft-config-files/agent_motion_config.yaml")
cfg = omegaconf.DictConfig(cfg)
rasterizer = build_rasterizer(cfg, dm)

model = LyftMultiModel(dm, cfg, rasterizer)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(cfg["model_params"]["weight_path"]))
model.to(device)


checkpoint_callback = ModelCheckpoint(
    verbose=0,
    monitor='val_loss',
    mode='min',
    prefix='lyfnet_',
)

trainer = Trainer(
    max_epochs=epochs,
    gradient_clip_val=cfg["model_params"]["gradient_clip_val"],
    logger=neptune_logger,
    checkpoint_callback=checkpoint_callback,
    # limit_val_batches=LIMIT_VAL_BATCHES,
    gpus=1
)

trainer.fit(model)


time_of_save = datetime.datetime.now().strftime("%d,%H")
torch.save(model.state_dict(), 'save/' + name_for_save + time_of_save + '.pt')