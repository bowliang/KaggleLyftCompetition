from catalyst import dl
from catalyst.dl import utils
from l5kit.data import ChunkedDataset, LocalDataManager, PERCEPTION_LABELS
from l5kit.dataset import EgoDataset, AgentDataset


import omegaconf
import os
from pytorch_lightning.core.lightning import LightningModule
import random
import segmentation_models_pytorch as smp
import torch 
# from torch import Tensor
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
import torch.nn as nn
from torch.utils.data import DataLoader



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np



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
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(gt, pred, avails):
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


class LyftMultiModel(LightningModule):

    def __init__(self, dm, cfg, rasterizer, num_modes=3):
        super().__init__()
        self.lr = cfg["model_params"]["lr"]
        self.weight_decay = cfg["model_params"]["weight_decay"]
        self.epochs = cfg["model_params"]["epochs"]

        self.dm = dm
        self.cfg = cfg
        self.rasterizer = rasterizer

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)


    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr= self.lr, betas= (0.9,0.999), 
                      weight_decay= self.weight_decay, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-5,
        )
        return [optimizer], [scheduler]


    def train_dataloader(self):
        train_cfg = omegaconf.OmegaConf.to_container((self.cfg.train_data_loader))
        train_zarr = ChunkedDataset(self.dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(self.cfg, train_zarr, self.rasterizer)
        subset = torch.utils.data.Subset(train_dataset, range(0, int(self.cfg["train_data_loader"]["training_percentage"] * len(train_dataset))))
        train_dataloader = DataLoader(subset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"],
                              drop_last=True)
        return train_dataloader


    def training_step(self, data, data_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = data["image"].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        targets = data["target_positions"].to(device)
        preds, confidences = self(x)
        loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)
        self.log('train_loss', loss)
        return {'loss': loss}

    def val_dataloader(self):
        validation_cfg = omegaconf.OmegaConf.to_container((self.cfg.val_data_loader))
        val_zarr = ChunkedDataset(self.dm.require(validation_cfg["key"])).open()
        val_dataset = AgentDataset(self.cfg, val_zarr, self.rasterizer)
        subset = torch.utils.data.Subset(val_dataset, range(0, int(self.cfg["val_data_loader"]["validation_percentage"] * len(val_dataset))))
        val_dataloader = DataLoader(subset,
                            shuffle=validation_cfg["shuffle"],
                            batch_size=validation_cfg["batch_size"],
                            num_workers=validation_cfg["num_workers"],
                            drop_last=True)
        return val_dataloader

    @torch.no_grad()
    def validation_step(self, data, data_idx):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = data["image"].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        targets = data["target_positions"].to(device)
        preds, confidences = self.forward(x)
        val_loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)
        self.log('val_loss', val_loss)
        return {'loss': val_loss}


# class LyftModel(torch.nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
        
#         self.backbone = smp.FPN(encoder_name="resnext50_32x4d", classes=1)
        
#         num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
#         num_in_channels = 3 + num_history_channels

#         self.backbone.encoder.conv1 = nn.Conv2d(
#             num_in_channels,
#              self.backbone.encoder.conv1.out_channels,
#             kernel_size= self.backbone.encoder.conv1.kernel_size,
#             stride= self.backbone.encoder.conv1.stride,
#             padding= self.backbone.encoder.conv1.padding,
#             bias=False,
#         )
#         backbone_out_features = 14

#         # X, Y coords for the future positions (output shape: Bx50x2)
#         num_targets = 2 * cfg["model_params"]["future_num_frames"]

#         self.head = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(in_features=14, out_features=4096),
#         )
#         self.backbone.segmentation_head = nn.Sequential(nn.Conv1d(56, 1, kernel_size=3, stride=2), nn.Dropout(0.2), nn.ReLU())
#         self.logit = nn.Linear(4096, out_features=num_targets)
#         self.logit_final = nn.Linear(128, 16)
#         self.num_preds = num_targets * 3
#         self.num_modes = 3
        
#     def forward(self, x):
#         x = self.backbone.encoder.conv1(x)
#         x = self.backbone.encoder.bn1(x)
#         x = self.backbone.encoder.relu(x)
#         x = self.backbone.encoder.maxpool(x)

#         x = self.backbone.encoder.layer1(x)
#         x = self.backbone.encoder.layer2(x)
#         x = self.backbone.encoder.layer3(x)
#         x = self.backbone.encoder.layer4(x)

#         x = self.backbone.decoder.p5(x)
#         x = self.backbone.decoder.seg_blocks[0](x)
#         x = self.backbone.decoder.merge(x)
#         x = self.backbone.segmentation_head(x)
#         x = self.backbone.encoder.maxpool(x)
       
#         x = torch.flatten(x, 1)
#         x = self.head(x)
#         x = self.logit(x)
#         x = x.permute(1, 0)
#         x = self.logit_final(x)


#         return x

# class LyftRunner(dl.SupervisedRunner):
#     def predict_batch(self, batch):
#         return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

#     def _handle_batch(self, batch):
#         x, y = batch['image'], batch['target_positions']
#         y_hat = self.model(x).view(y.shape)
#         target_availabilities = batch["target_availabilities"].unsqueeze(-1)
#         criterion = torch.nn.MSELoss(reduction="none")
        
#         loss = criterion(y_hat, y)
#         loss = loss * target_availabilities
#         loss = loss.mean()
#         self.batch_metrics.update(
#             {"loss": loss}
#         )

