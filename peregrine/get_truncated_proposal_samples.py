import sys
from datetime import datetime
import pickle
import swyft.lightning as sl
from config_utils_snpe import read_config, init_config
from simulator_utils_snpe import init_simulator
from sbi.diagnostics.tarp import run_tarp
from sbi.analysis import plot_tarp

from sbi.inference import SNPE
import torch
import torch.nn.functional as F
import torch.distributions as dist

# For parallelisation
import subprocess
import logging
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import entropy, gaussian_kde
from sbi.diagnostics.tarp import run_tarp

import pandas as pd
import numpy as np
import json
import glob

class SineDistribution(dist.Distribution):
    arg_constraints = {}

    def __init__(self, device='cpu', validate_args=None):
        super().__init__(validate_args=validate_args)
        self.device = device

    @property
    def support(self):
        return dist.constraints.interval(0.0, torch.pi)

    def to(self, device):
        return SineDistribution(device=device)

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape, device=self.device)
        return torch.acos(1 - 2 * u)

    def log_prob(self, x):
        x = x.to(self.device)
        inside_support = (x >= 0.0) & (x <= torch.pi)
        log_probs = torch.where(
            inside_support,
            torch.log(0.5 * torch.sin(x)),
            torch.tensor(float("-inf"), device=self.device)
        )
        return log_probs

class CosineDistribution(dist.Distribution):
    arg_constraints = {}

    def __init__(self, device='cpu', validate_args=None):
        super().__init__(validate_args=validate_args)
        self.device = device

    @property
    def support(self):
        return dist.constraints.interval(-torch.pi / 2, torch.pi / 2)

    def to(self, device):
        return CosineDistribution(device=device)

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape, device=self.device)
        return torch.asin(2 * u - 1)

    def log_prob(self, x):
        x = x.to(self.device)
        inside_support = (x >= -torch.pi / 2) & (x <= torch.pi / 2)
        log_probs = torch.where(
            inside_support,
            torch.log(0.5 * torch.cos(x)),
            torch.tensor(float("-inf"), device=self.device)
        )
        return log_probs

def move_dist_to_device(d, device):
    """Move a distribution's parameters to the specified device."""
    if isinstance(d, dist.Uniform):
        return dist.Uniform(d.low.to(device), d.high.to(device), validate_args=False)
    elif isinstance(d, dist.Normal):
        return dist.Normal(d.loc.to(device), d.scale.to(device), validate_args=False)
    elif isinstance(d, dist.TransformedDistribution):
        base = move_dist_to_device(d.base_dist, device)
        transforms = d.transforms  # transforms don't need device change
        return dist.TransformedDistribution(base, transforms)
    elif hasattr(d, 'to'):
        return d.to(device)
    else:
        raise NotImplementedError(f"Device transfer not implemented for distribution type: {type(d)}")

class JointPriorTensor(dist.Distribution):
    arg_constraints = {}

    def __init__(self, priors, keys_order=None, device="cpu"):
        self.device = device
        self.priors = {k: move_dist_to_device(v, device) for k, v in priors.items()}
        self.keys_order = keys_order or list(priors.keys())

        self.low = torch.tensor(
            [self.priors[k].support.lower_bound for k in self.keys_order],
            device=self.device
        ).float()

        self.high = torch.tensor(
            [self.priors[k].support.upper_bound for k in self.keys_order],
            device=self.device
        ).float()

        super().__init__()

    def to(self, device):
        return JointPriorTensor(self.priors, self.keys_order, device)

    @property
    def support(self):
        return dist.constraints.interval(self.low, self.high)

    def sample(self, sample_shape=torch.Size()):
        samples_list = []
        for key in self.keys_order:
            sample_val = self.priors[key].sample(sample_shape)
            if sample_val.ndim > len(sample_shape):
                sample_val = sample_val.squeeze(-1)
            samples_list.append(sample_val)
        return torch.stack(samples_list, dim=-1)

    def log_prob(self, samples_tensor):
        log_probs = []
        for i, key in enumerate(self.keys_order):
            sample_val = samples_tensor[..., i]
            log_prob_val = self.priors[key].log_prob(sample_val)
            log_probs.append(log_prob_val)
        return sum(log_probs)

def move_priors_to_device(priors, device):
    new_priors = {}
    for key, p in priors.items():
        if isinstance(p, dist.Uniform):
            new_priors[key] = dist.Uniform(
                p.low.to(device),
                p.high.to(device)
            )
        elif isinstance(p, dist.Normal):
            new_priors[key] = dist.Normal(
                p.loc.to(device),
                p.scale.to(device)
            )
        elif isinstance(p, SineDistribution) or isinstance(p, CosineDistribution):
            new_priors[key] = p.to(device)
    return new_priors

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Reading config file"
    )
    print(f"Config: {args[0]}")
    snpe_parser = read_config(args)
    conf = init_config(snpe_parser, args)
    logging.basicConfig(
        filename=f"{conf['zarr_params']['store_path']}/log_{conf['zarr_params']['run_id']}.log",
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    simulator = init_simulator(conf)
    bounds = None
    if conf["snpe"]["generate_obs"]:
        obs = simulator.generate_observation()
        logging.warning(
            f"Overwriting observation file: {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
        )
        with open(
            f"{conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
            "wb",
        ) as f:
            pickle.dump(obs, f)
    else:
        observation_path = conf["snpe"]["obs_path"]
        with open(observation_path, "rb") as f:
            obs = pickle.load(f)
        subprocess.run(
            f"cp {observation_path} {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
            shell=True,
        )
    logging.info(
        f"Observation loaded and saved in {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
    )
    obs = sl.Sample(
        {key: obs[key] for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w"]}
    )
    
    # Define a fixed ordering for the parameters.
    priors = {
        "mass_ratio": dist.Uniform(torch.tensor([0.125]), torch.tensor([1.0])),
        "chirp_mass": dist.Uniform(torch.tensor([25.0]), torch.tensor([100.0])),
        "theta_jn": SineDistribution(), 
        "phase": dist.Uniform(torch.tensor([0.0]), torch.tensor([6.28318])),
        "tilt_1": SineDistribution(),
        "tilt_2": SineDistribution(),
        "a_1": dist.Uniform(torch.tensor([0.05]), torch.tensor([1.0])),
        "a_2": dist.Uniform(torch.tensor([0.05]), torch.tensor([1.0])),
        "phi_12": dist.Uniform(torch.tensor([0.0]), torch.tensor([6.28318])),
        "phi_jl": dist.Uniform(torch.tensor([0.0]), torch.tensor([6.28318])),
        "luminosity_distance": dist.Uniform(torch.tensor([100.0]), torch.tensor([1500.0])),
        "dec": CosineDistribution(),  
        "ra": dist.Uniform(torch.tensor([0.0]), torch.tensor([6.28318])),
        "psi": dist.Uniform(torch.tensor([0.0]), torch.tensor([3.14159])),
        "geocent_time": dist.Uniform(torch.tensor([-0.1]), torch.tensor([0.1])),
    }

    order = [
        "mass_ratio",
        "chirp_mass",
        "theta_jn",
        "phase",
        "tilt_1",
        "tilt_2",
        "a_1",
        "a_2",
        "phi_12",
        "phi_jl",
        "luminosity_distance",
        "dec",
        "ra",
        "psi",
        "geocent_time",
    ] #same ordering as int / ext priors I believe - do check though
    
    device = 'cuda'

    priors_on_device = move_priors_to_device(priors, device)
    joint_prior = JointPriorTensor(priors_on_device, keys_order=list(priors.keys()), device="cuda")

    true_params = [
        0.8857620985418904,
        32.136969061169324,
        0.44320777946320117,
        5.089358282766109,
        1.4974326044527126,
        1.1019600169566186,
        0.9701993491043245,
        0.8117959745751914,
        6.220246980963511,
        1.884805935473119,
        900,
        0.07084716171380845,
        5.555599820502261,
        1.0995170458005799,
        0.0
    ]

    #turn true_params into [1,15] torch tensor
    true_params = torch.tensor([true_params])
    round_id = 4
    
    def pad_to_width(t, target_width, i):
        current_width = t.shape[i]
        if current_width < target_width:
            pad_amount = target_width - current_width
            return F.pad(t, (0, pad_amount))
        return t   

    def pad_to_length(t, target_length, i):
        current_length = t.shape[i]
        if current_length < target_length:
            pad_amount = target_length - current_length
            return F.pad(t, (0,0,0,pad_amount))
        return t  

    obs = (
            {key: torch.tensor(obs[key]) for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w"]}
        )

    obs["d_t"] = pad_to_length(obs["d_t"], 6, 0) 
    obs["n_t"] = pad_to_length(obs["n_t"], 6, 0) 
    obs["d_f"] = pad_to_width(obs["d_f"], 8192, 1) 
    obs["d_f_w"] = pad_to_width(obs["d_f_w"], 8192, 1)
    obs["n_f"] = pad_to_width(obs["n_f"], 8192, 1)
    obs["n_f_w"] = pad_to_width(obs["n_f_w"], 8192, 1) 

    obs = [obs[key] for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w"]]
    obs = torch.cat(obs, dim=1)
    obs = obs.to('cuda')
        
    loaded_posterior = torch.load(f'/data/kn405/Code/peregrine/peregrine/snpe_no_weight_round_4_posterior.pt')
    posterior_samples = loaded_posterior.sample_batched(torch.Size([100000]), x=obs)
    posterior_samples = posterior_samples.cpu()
    np.save('/data/kn405/Code/peregrine/peregrine/proposal_samples_truncated_prior.npy', posterior_samples)

    def compute_hpd(samples, credible_mass=0.99):
        """
        Compute the Highest Posterior Density (HPD) interval for a 1D array of samples.
        """
        sorted_samples = np.sort(samples)
        n = len(samples)
        interval_idx_inc = int(np.floor(credible_mass * n))
        n_intervals = n - interval_idx_inc
        interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError("Not enough samples to compute HPD.")

        min_idx = np.argmin(interval_width)
        hpd_min = sorted_samples[min_idx]
        hpd_max = sorted_samples[min_idx + interval_idx_inc]

        return hpd_min, hpd_max

    def compute_hpd_intervals(posterior_samples, credible_mass=0.99):
        """
        Compute HPD intervals for each parameter in posterior_samples.
        posterior_samples: numpy array of shape [n_samples, n_parameters]
        """
        n_params = posterior_samples.shape[1]
        hpd_intervals = []

        for i in range(n_params):
            hpd = compute_hpd(posterior_samples[:, i], credible_mass=credible_mass)
            hpd_intervals.append(hpd)

        return np.array(hpd_intervals)  # shape [n_params, 2]

    # Example usage:
    # posterior_samples is a NumPy array of shape [1000, 15]
    hpd_99 = compute_hpd_intervals(posterior_samples, credible_mass=0.99)



