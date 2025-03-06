import numpy as np
import os
import torch
from torch import nn
from torch.functional import F
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import swyft.lightning as sl
from sbi.inference import SNPE
from sbi.neural_nets.net_builders import build_nsf, build_maf

class InferenceNetwork(sl.SwyftModule):
    def __init__(self, conf):
        super().__init__()
        self.one_d_only = conf["snpe"]["1d_only"]
        self.batch_size = conf["hparams"]["training_batch_size"]
        self.noise_shuffling = conf["snpe"]["shuffling"]
        self.num_params = len(conf["priors"]["int_priors"].keys()) + len(
            conf["priors"]["ext_priors"].keys()
        )
        self.marginals = conf["snpe"]["marginals"]
        self.unet_t = Unet(
            n_in_channels=len(conf["waveform_params"]["ifo_list"]),
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(8, 8, 8, 8),
        )
        self.unet_f = Unet(
            n_in_channels=2 * len(conf["waveform_params"]["ifo_list"]),
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(2, 2, 2, 2),
        )

        self.flatten = nn.Flatten(1)
        self.linear_t = LinearCompression()
        self.linear_f = LinearCompression()

        if not self.one_d_only:
            self.linear_t_2d = LinearCompression_2d()
            self.linear_f_2d = LinearCompression_2d()

        self.optimizer_init = sl.AdamOptimizerInit(lr=conf["hparams"]["learning_rate"])

    def forward(self, x):
        
        n_t = x[:, :3, 24576:32768]
        n_f_w = x[:, :, 40960:45057]
        d_t = x[:, :3, 0:8192]
        d_f_w = x[:, :, 16384:20481]
        
        training_batch_size = n_t.size(0)

        if self.noise_shuffling and d_t.size(0) != 1:
            noise_shuffling = torch.randperm(training_batch_size)
            d_t = d_t + n_t[noise_shuffling]
            d_f_w = d_f_w + n_f_w[noise_shuffling]
        else:
            d_t = d_t + n_t
            d_f_w = d_f_w + n_f_w
                     
        d_t = self.unet_t(d_t)
        d_f_w = self.unet_f(d_f_w)

        if not self.one_d_only:
            features_t = self.linear_t(self.flatten(d_t))
            features_f = self.linear_f(self.flatten(d_f_w))
            features = torch.cat([features_t, features_f], dim=1)
            features = features_t
            features_t_2d = self.linear_t_2d(self.flatten(d_t))
            features_f_2d = self.linear_f_2d(self.flatten(d_f_w))
            features_2d = torch.cat([features_t_2d, features_f_2d], dim=1)
            features_2d = features_t_2d
            return features, features_2d
        else:
            features_t = self.linear_t(self.flatten(d_t))
            features_f = self.linear_f(self.flatten(d_f_w))
            features = torch.cat([features_t, features_f], dim=1)
            features = features_t
            return features


def init_network(conf: dict):
    """
    Initialise the network with the settings given in a loaded config dictionary
    Args:
      conf: dictionary of config options, output of init_config
    Returns:
      Pytorch lightning network class
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> snpe_parser = read_config(sysargs)
    >>> conf = init_config(snpe_parser, sysargs, sim=True)
    >>> network = init_network(conf)
    """
    network = InferenceNetwork(conf)
    return network


def setup_zarr_store(
    conf: dict,
    simulator,
    round_id: int = None,
    coverage: bool = False,
    n_sims: int = None,
):
    """
    Initialise or load a zarr store for saving simulations
    Args:
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
      coverage: specifies if store should be used for coverage sims
      n_sims: number of simulations to initialise store with
    Returns:
      Zarr store object
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> snpe_parser = read_config(sysargs)
    >>> conf = init_config(snpe_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    """
    zarr_params = conf["zarr_params"]
    if zarr_params["use_zarr"]:
        chunk_size = zarr_params["chunk_size"]
        if n_sims is None:
            if "nsims" in zarr_params.keys():
                n_sims = zarr_params["nsims"]
            else:
                n_sims = zarr_params["sim_schedule"][round_id - 1]
        shapes, dtypes = simulator.get_shapes_and_dtypes()
        shapes.pop("n")
        dtypes.pop("n")
        store_path = zarr_params["store_path"]
        if round_id is not None:
            if coverage:
                store_dir = f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R{round_id}"
            else:
                store_dir = (
                    f"{store_path}/simulations_{zarr_params['run_id']}_R{round_id}"
                )
        else:
            if coverage:
                store_dir = (
                    f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R1"
                )
            else:
                store_dir = f"{store_path}/simulations_{zarr_params['run_id']}_R1"

        store = sl.ZarrStore(f"{store_dir}")
        store.init(N=n_sims, chunk_size=chunk_size, shapes=shapes, dtypes=dtypes)
        return store
    else:
        return None


def setup_dataloader(store, simulator, conf: dict, round_id: int = None):
    """
    Initialise a dataloader to read in simulations from a zarr store
    Args:
      store: zarr store to load from, output of setup_zarr_store
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
    Returns:
      (training dataloader, validation dataloader), trainer directory
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> snpe_parser = read_config(sysargs)
    >>> conf = init_config(snpe_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    >>> train_data, val_data, trainer_dir = setup_dataloader(store, simulator, conf)
    """
    if round_id is not None:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R{round_id}"
    else:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R1"
    if not os.path.isdir(trainer_dir):
        os.mkdir(trainer_dir)
    hparams = conf["hparams"]
    if conf["snpe"]["resampler"]:
        resampler = simulator.get_resampler(targets=conf["snpe"]["noise_targets"])
    else:
        resampler = None
    train_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["training_batch_size"]),
        idx_range=[0, int(hparams["train_data"] * len(store.data.z_int))],
        on_after_load_sample=resampler,
    )
    val_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["validation_batch_size"]),
        idx_range=[
            int(hparams["train_data"] * len(store.data.z_int)),
            len(store.data.z_int) - 1,
        ],
        on_after_load_sample=None,
    )
    return train_data, val_data, trainer_dir


def setup_density_estimator(conf: dict, dummy_theta, dummy_x):
    """
    Initialise a pytorch lightning trainer and relevant directories
    Args:
      trainer_dir: location for the training logs
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Swyft lightning trainer instance
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> snpe_parser = read_config(sysargs)
    >>> conf = init_config(snpe_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator, 1)
    >>> train_data, val_data, trainer_dir = setup_dataloader(store, simulator, conf, 1)
    >>> trainer = setup_trainer(trainer_dir, conf, 1)
    """

    device_params = conf["device_params"]
    hparams = conf["hparams"]

    embedding_net = init_network(conf)
    density_estimator = build_maf(
        dummy_theta,
        dummy_x,
        embedding_net = embedding_net,
    )
    # current_lr = optimizer.param_groups[0]["lr"]  
    return density_estimator

def save_bounds(bounds, conf: dict, round_id: int):
    """
    Save bounds from a particular round
    Args:
      bounds: unpacked swyft bounds object
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    np.savetxt(
        f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id}.txt",
        bounds,
    )


def load_bounds(conf: dict, round_id: int):
    """
    Load bounds from a particular round
    Args:
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Bounds object with ordering defined by the param idxs in the config
    """
    if round_id == 1:
        return None
    else:
        bounds = np.loadtxt(
            f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id - 1}.txt"
        )
        return bounds


def save_coverage(coverage, conf: dict, round_id: int):
    """
    Save coverage samples from a particular round
    Args:
      coverage: swyft coverage object instance
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if not os.path.isdir(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
    ):
        os.mkdir(
            f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
        )
    with open(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}/coverage_R{round_id}",
        "wb",
    ) as p:
        pickle.dump(coverage, p)


# 1D Unet implementation below
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        mid_channels=None,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(down_sampling), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_signal_length = x2.size()[2] - x1.size()[2]

        x1 = F.pad(
            x1, [diff_signal_length // 2, diff_signal_length - diff_signal_length // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_out_channels,
        sizes=(16, 32, 64, 128, 256),
        down_sampling=(2, 2, 2, 2),
    ):
        super(Unet, self).__init__()
        self.inc = DoubleConv(n_in_channels, sizes[0])
        self.down1 = Down(sizes[0], sizes[1], down_sampling[0])
        self.down2 = Down(sizes[1], sizes[2], down_sampling[1])
        self.down3 = Down(sizes[2], sizes[3], down_sampling[2])
        self.down4 = Down(sizes[3], sizes[4], down_sampling[3])
        self.up1 = Up(sizes[4], sizes[3])
        self.up2 = Up(sizes[3], sizes[2])
        self.up3 = Up(sizes[2], sizes[1])
        self.up4 = Up(sizes[1], sizes[0])
        self.outc = OutConv(sizes[0], n_out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        f = self.outc(x)
        return f


class LinearCompression(nn.Module):
    def __init__(self):
        super(LinearCompression, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(16),
        )

    def forward(self, x):
        return self.sequential(x)


class LinearCompression_2d(nn.Module):
    def __init__(self):
        super(LinearCompression_2d, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(128),
        )

    def forward(self, x):
        return self.sequential(x)