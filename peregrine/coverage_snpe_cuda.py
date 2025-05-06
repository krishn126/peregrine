print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: SNPE Coverage Test
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey, K.Nanavati
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
import numpy as np
from datetime import datetime
import swyft.lightning as sl
from config_utils_snpe import read_config, init_config
from simulator_utils_snpe import init_simulator, simulate
from inference_utils_snpe_custom import (
    setup_zarr_store,
    setup_dataloader,
    setup_density_estimator,
    load_bounds,
    setup_scheduler,
)
import subprocess
import psutil
import time
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import subprocess
import psutil
import logging
import pickle
import torch.distributions as dist
from sbi.diagnostics.tarp import run_tarp
from sbi.analysis.plot import plot_tarp

ranges = [
        (0.125, 1.0),  # mass_ratio
        (30.0, 35.0),  # chirp_mass
        (0.0, 3.14159/2),  # theta_jn
        (0.0, 6.28318),  # phase
        (0.0, 3.14159),  # tilt_1
        (0.0, 3.14159),  # tilt_2
        (0.05, 1.0),  # a_1
        (0.05, 1.0),  # a_2
        (0.0, 6.28318),  # phi_12
        (0.0, 6.28318/2),  # phi_jl
        (600.0, 1200.0),  # luminosity_distance
        (-0.05, 0.2),  # dec
        (5.4, 5.8),  # ra
        (0.0, 3.14159),  # psi
        (-0.005, 0.005),  # geocent_time
    ]

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
    for round_id in range(1, int(conf["snpe"]["num_rounds"]) + 1):
        # Initialise the zarr store to save the simulations
        start_time = datetime.now()
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Initialising zarrstore for round {round_id}"
        )
        store = setup_zarr_store(conf, simulator, round_id=round_id)
        logging.info(f"Starting simulations for round {round_id}")
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Simulating data for round {round_id}"
        )
        if conf["zarr_params"]["run_parallel"]:
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Running in parallel - spawning processes"
            )
            processes = []
            if conf["zarr_params"]["njobs"] == -1:
                njobs = psutil.cpu_count(logical=True)
            elif conf["zarr_params"]["njobs"] > psutil.cpu_count(logical=False):
                njobs = psutil.cpu_count(logical=True)
            else:
                njobs = conf["zarr_params"]["njobs"]
            for job in range(njobs):
                p = subprocess.Popen(
                    [
                        "python",
                        "run_parallel_snpe.py",
                        f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.txt",
                        str(round_id),
                        f"proposal_samples_round_{round_id}.npy"
                    ]
                )
                processes.append(p)
            for p in processes:
                p.wait()
        else:
            bounds = load_bounds(conf, round_id)
            simulator = init_simulator(conf, bounds)
            simulate(simulator, store, conf)
        logging.info(f"Simulations for round {round_id} completed")
        # Initialise data loader for training
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Setting up dataloaders for round {round_id}"
        )
        train_data, val_data, trainer_dir = setup_dataloader(
            store, simulator, conf, round_id
        )

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
                "dec": CosineDistribution(),  # Assuming it already has event_shape=(1,)
                "ra": dist.Uniform(torch.tensor([0.0]), torch.tensor([6.28318])),
                "psi": dist.Uniform(torch.tensor([0.0]), torch.tensor([3.14159])),
                "geocent_time": dist.Uniform(torch.tensor([-0.1]), torch.tensor([0.1])),
            }

    # Define a fixed ordering for the parameters.
    
    # Create a joint prior distribution    
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
    ]

    label_order = [
        r"Mass Ratio, $q$",                  # mass_ratio
        r"Chirp Mass, $\mathcal{M}_{\mathrm{C}}$",        # chirp_mass (chirp mass symbol, often curly M)
        r"$\theta_{\mathrm{jn}}$",  # theta_jn
        r"Phase, $\phi$",               # phase
        r"$\theta_1$",           # tilt_1
        r"$\theta_2$",           # tilt_2
        r"$a_1$",                # a_1
        r"$a_2$",                # a_2
        r"$\phi_{12}$",          # phi_12
        r"$\phi_{\mathrm{jl}}$", # phi_jl
        r"Luminosity Distance, $d_L$",                # luminosity_distance
        r"Declination, $\delta$",             # dec (declination)
        r"Right Ascension, $\alpha$",             # ra (right ascension)
        r"Polarization Angle, $\psi$",               # psi (polarization angle)
        r"Geocent time, $t_0$",                # geocent_time
    ]

    device = 'cuda'

    priors_on_device = move_priors_to_device(priors, device)
    joint_prior = JointPriorTensor(priors_on_device, keys_order=list(priors.keys()), device="cuda")
    
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

    def get_theta(d):
        d =   (
                {key: torch.tensor(d[key]) for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w", "z_total"]}
            ) 
        return d["z_total"]

    def get_data(d):
        d =   (
                {key: torch.tensor(d[key]) for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w", "z_total"]}
            )       
        d["d_t"] = pad_to_length(d["d_t"], 6, 1) 
        d["n_t"] = pad_to_length(d["n_t"], 6, 1)
        d["d_f"] = pad_to_width(d["d_f"], 8192, 2)
        d["d_f_w"] = pad_to_width(d["d_f_w"], 8192, 2)
        d["n_f"] = pad_to_width(d["n_f"], 8192, 2)
        d["n_f_w"] = pad_to_width(d["n_f_w"], 8192, 2)

        d = [d[key] for key in ["d_t", "d_f", "d_f_w", "n_t", "n_f", "n_f_w"]]
        d = torch.cat(d, dim=2)

        return d
    
    loaded_density_estimator = torch.load('/data/kn405/Code/peregrine/peregrine/tsnpe_de.pt')
    loaded_posterior = torch.load('/data/kn405/Code/peregrine/peregrine/tsnpe_posterior.pt')
    limit = 1

    for i, sample in enumerate(train_data):
        if i > limit:
            break
        theta_train = get_theta(sample)
        x_train = get_data(sample)
    
    #Manual coverage test
    posterior_samples = []
    for j in range(100):
            tester = x_train[j, :, :]
            tester = tester.to('cuda')
            posterior_samples.append(loaded_posterior.sample_batched(torch.Size([1000]), x=tester))
    
    
    def compute_coverage_per_param(num_sims, true_parameters, credibility_levels=torch.linspace(0.05, 0.95, 15)):
        _, num_parameters = true_parameters.shape
        num_levels = credibility_levels.shape[0]
        empirical_coverages = torch.zeros(num_parameters, num_levels)

        # Sort posterior samples along the sample axis
        for j in range(num_sims):
            post_samples = posterior_samples[j]
            post_samples = post_samples.squeeze(1)
            sorted_samples, _ = torch.sort(post_samples, dim=0)
            num_samples, _ = sorted_samples.shape

            for i, level in enumerate(credibility_levels):
                lower_idx = int((1 - level) / 2 * num_samples)
                upper_idx = int((1 + level) / 2 * num_samples)

                lower_bounds = sorted_samples[lower_idx, :]
                upper_bounds = sorted_samples[upper_idx, :]
                # Compute coverage per parameter
                for k in range(num_parameters):
                    if true_parameters[j, k] >= lower_bounds[k] and true_parameters[j, k] <= upper_bounds[k]:
                        empirical_coverages[k, i] += 1
            
        empirical_coverages /= num_sims

        return credibility_levels.numpy(), empirical_coverages.numpy()

    def plot_coverage(credibility_levels, empirical_coverages):
        num_parameters = empirical_coverages.shape[0]
        plt.figure(figsize=(15, 8))
        # Plot coverage for each parameter
        for i in range(num_parameters):
            ax = plt.subplot(5, 3, i + 1)
            ax.set_title(f"{label_order[i]}", fontsize=10)
            plt.plot(credibility_levels, empirical_coverages[i, :], marker='o', linestyle='-', alpha=0.7, label=f"Param {order[i]}")
            plt.plot([0, 1], [0, 1], 'k--', label="Ideal Calibration") # Reference y=x line for perfect calibration
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"/data/kn405/Code/peregrine/posterior_plots/empirical_coverage_tests.png", dpi=300, bbox_inches='tight')

    # Compute and plot empirical coverage
 
    credibility_levels, empirical_coverages = compute_coverage_per_param(100, theta_train)
    plot_coverage(credibility_levels, empirical_coverages)

    #TARP test - not working? 
    # ecp, alpha = run_tarp(
    #     theta_train,
    #     x_train,
    #     loaded_posterior,
    #     references=None,  # will be calculated automatically.
    #     num_posterior_samples=1000,
    # )

    # plot_tarp(ecp, alpha)
    # plt.savefig(f"/data/kn405/Code/peregrine/posterior_plots/tarp_plot.png", dpi=300, bbox_inches='tight')
