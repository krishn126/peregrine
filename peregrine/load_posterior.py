print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: SNPE Inference
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey, K.Nanavati
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
from datetime import datetime
import pickle
import swyft.lightning as sl
from config_utils_snpe import read_config, init_config
from simulator_utils_snpe import init_simulator

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

import pandas as pd
import numpy as np
import json
import glob

class SineDistribution(dist.Distribution):
    """
    Custom sine-distributed probability distribution over [0, pi],
    defined by p(x) = (1/2) * sin(x).
    """   
    support = dist.constraints.interval(torch.tensor([0.0]), torch.pi)  # Support is [0, pi]

    def __init__(self, validate_args=None):
        super().__init__(validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Uses inverse CDF sampling: x = arccos(1 - U), where U ~ Uniform(0,1)
        """
        u = torch.rand(sample_shape)
        return torch.acos(1 - u)  # Returns samples in [0, pi]

    def log_prob(self, x):
        """
        Log probability of the sine distribution: log( (1/2) * sin(x) )
        """
        inside_support = (x >= torch.tensor([0.0])) & (x <= torch.pi)
        log_probs = torch.where(
            inside_support,
            torch.log(torch.tensor([0.5]) * torch.sin(x)),  # log(p(x))
            torch.tensor(float("-inf"))  # Log prob is -inf outside support
        )
        return log_probs

class CosineDistribution(dist.Distribution):
    """
    Custom cosine-distributed probability distribution over [-pi/2, pi/2],
    defined by p(x) = (1/2) * cos(x).
    """
    support = dist.constraints.interval(-torch.pi / 2, torch.pi / 2)  # Support is [-π/2, π/2]

    def __init__(self, validate_args=None):
        super().__init__(validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Uses inverse CDF sampling: x = arcsin(2U - 1), where U ~ Uniform(0,1)
        """
        u = torch.rand(sample_shape)
        return torch.asin(2 * u - 1)  # Returns samples in [-π/2, π/2]

    def log_prob(self, x):
        """
        Log probability of the cosine distribution: log( (1/2) * cos(x) )
        """
        inside_support = (x >= -torch.pi / 2) & (x <= torch.pi / 2)
        log_probs = torch.where(
            inside_support,
            torch.log(torch.tensor([0.5]) * torch.cos(x)),  # log(p(x))
            torch.tensor(float("-inf"))  # Log prob is -inf outside support
        )
        return log_probs

class JointPriorTensor(dist.Distribution):
    def __init__(self, priors, keys_order=None):
        """
        Args:
            priors (dict): A dictionary of individual priors.
            keys_order (list, optional): An ordered list of keys to define
                the order in which samples are stacked. If None, uses
                list(priors.keys()).
        """
        self.priors = priors
        if keys_order is None:
            keys_order = list(priors.keys())
        self.keys_order = keys_order
        super().__init__()
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from each individual prior and stack the results into a single tensor.
        
        Returns:
            Tensor of shape sample_shape + (num_priors,)
        """
        samples_list = []
        for key in self.keys_order:
            # Sample from the individual prior.
            sample_val = self.priors[key].sample(sample_shape)
            # If the sample has an extra dimension (e.g., shape (..., 1)), squeeze it.
            if sample_val.ndim > len(sample_shape):
                sample_val = sample_val.squeeze(-1)
            samples_list.append(sample_val)
        # Stack along the last dimension so that each sample is a vector.
        samples_tensor = torch.stack(samples_list, dim=-1)
        return samples_tensor
    
    def log_prob(self, samples_tensor):
        """
        Computes the joint log probability by splitting the tensor and summing
        individual log probabilities.
        
        Args:
            samples_tensor (Tensor): A tensor of shape sample_shape + (num_priors,)
            
        Returns:
            A tensor of shape sample_shape with the joint log probability.
        """
        log_probs = []
        for i, key in enumerate(self.keys_order):
            # Extract the sample corresponding to the i-th prior.
            sample_val = samples_tensor[..., i]
            log_prob_val = self.priors[key].log_prob(sample_val)
            log_probs.append(log_prob_val)
        # Sum the log probabilities (since the priors are independent).
        total_log_prob = sum(log_probs)
        return total_log_prob

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
            "dec": CosineDistribution(),  # Assuming it already has event_shape=(1,)
            "ra": dist.Uniform(torch.tensor([0.0]), torch.tensor([6.28318])),
            "psi": dist.Uniform(torch.tensor([0.0]), torch.tensor([3.14159])),
            "geocent_time": dist.Uniform(torch.tensor([-0.1]), torch.tensor([0.1])),
        }
    
    joint_prior = JointPriorTensor(priors)

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
    
    loaded_density_estimator = torch.load('/data/kn405/Code/peregrine/peregrine/density_estimator.pt')
    loaded_posterior = torch.load('/data/kn405/Code/peregrine/peregrine/posterior.pt')
    posterior_samples = loaded_posterior.sample_batched(torch.Size([100000]), x=obs)

    #save posterior samples
    np.save('/data/kn405/Code/peregrine//peregrine/posterior_samples.npy', posterior_samples)

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
    # # Plot posterior
    lrs = pd.read_pickle('/data/kn405/Code/peregrine/peregrine/logratios_R7')

    def load_dynesty(folder):
        with open(glob.glob(f"{folder}/dynesty_result.json")[0], "r") as f:
            dynesty = json.load(f)
        return dynesty["posterior"]["content"]

    dynesty_posterior = load_dynesty("/data/kn405/Code/peregrine_snpe/peregrine/peregrine")
    dynesty_posterior = np.array([dynesty_posterior[key] for key in order]).T
    

    def crb_from_posterior_samples(posterior_samples):
        """
        Approximate the CRB using posterior samples by calculating the inverse of the variance.

        Args:
            posterior_samples: Tensor of shape [N_samples, D_params] containing posterior samples.

        Returns:
            CRB: Cramér-Rao Bound as an approximation.
        """
        # Compute the covariance matrix from posterior samples
        covariance_matrix = np.cov(posterior_samples.T)

        # Compute the CRB as the trace of the inverse of the covariance matrix
        crb_approx = np.trace(np.log(np.linalg.inv(covariance_matrix)))

        return crb_approx
    
    # posterior_samples = posterior_samples.squeeze(1)
    # CRB = crb_from_posterior_samples(posterior_samples)
    # print(f"Approximate Cramér-Rao Bound (CRB): {CRB}")

    # fig = plt.figure(figsize=(15, 8))
    # for idx in range(15):
    #     ax = plt.subplot(5, 3, idx + 1)
    #     ax.set_title(f"{order[idx]}")
    #     logratios = lrs.logratios[:, idx]
    #     params = lrs.params[:, idx, 0]
    #     weights1 = np.ones_like(posterior_samples[:,0,idx])
    #     weights2 = np.exp(logratios.numpy())
    #     plt.hist([posterior_samples[:,0,idx].numpy(), params], weights = [weights1, weights2], range=ranges[idx], bins=100, density=True, alpha=0.7)
    #     plt.axvline(x=true_params[idx], linestyle='--')
    # fig.suptitle("NPE vs TMNRE", fontsize=20)
    # plt.tight_layout()
    # blue_line = mlines.Line2D([], [], color='blue', label='SNPE')
    # orange_line = mlines.Line2D([], [], color='orange', label='TMNRE')
    # fig.legend(handles=[blue_line, orange_line], loc="upper right", fontsize=10)
    
    # fig = corner.corner(posterior_samples[:,0,:].numpy(), color='blue', range=ranges, labels=order, hist_kwargs={"density": True})
    # corner.corner(dynesty_posterior, color='red', fig=fig, range=ranges, hist_kwargs={"density": True})
    # fig.suptitle('PROVISIONAL: SNPE vs Dynesty', fontsize=50)
    # blue_line = mlines.Line2D([], [], color='blue', label='SNPE')
    # red_line = mlines.Line2D([], [], color='red', label='Dynesty')
    # fig.legend(handles=[blue_line, red_line], loc="upper right", fontsize=40) # Add the legend

    #corner plot of posteriors
    # plt.savefig(f"/data/kn405/Code/peregrine/posterior_plots/snpe_vs_tmrne.png", dpi=300, bbox_inches='tight')

    def js_divergence(samples_p, samples_q, num_points=1000):
    # Define a common evaluation range
        min_val = min(samples_p.min(), samples_q.min()) 
        max_val = max(samples_p.max(), samples_q.max())
        eval_points = np.linspace(min_val, max_val, num_points)
        
        # Estimate densities using KDE
        kde_p = gaussian_kde(samples_p)
        kde_q = gaussian_kde(samples_q)
        
        pdf_p = kde_p(eval_points)
        pdf_q = kde_q(eval_points)
        
        # Normalize to ensure they sum to 1 (avoid numerical issues)
        pdf_p /= pdf_p.sum()
        pdf_q /= pdf_q.sum()
        
        # Compute mixed distribution
        pdf_m = 0.5 * (pdf_p + pdf_q)
        
        # Compute KL divergences and JS divergence
        kl_p_m = entropy(pdf_p, pdf_m)  # KL(P || M)
        kl_q_m = entropy(pdf_q, pdf_m)  # KL(Q || M)
        
        js_div = 0.5 * (kl_p_m + kl_q_m)
        return js_div
    
    def js_divergence_hist(samples_p, samples_q, weights, bins=50):
        hist_p, bin_edges = np.histogram(samples_p, bins=bins, density=True)
        hist_q, _ = np.histogram(samples_q, weights=weights, bins=bin_edges, density=True)
        
        # Convert to probabilities
        hist_p += 1e-10  # Avoid log(0)
        hist_q += 1e-10
        
        hist_p /= hist_p.sum()
        hist_q /= hist_q.sum()
        
        hist_m = 0.5 * (hist_p + hist_q)
        
        kl_p_m = entropy(hist_p, hist_m)
        kl_q_m = entropy(hist_q, hist_m)
        
        js_div = 0.5 * (kl_p_m + kl_q_m)
        return js_div

    # Example usage with posterior samples
    js_div_dyn = 0.0
    js_div_per = 0.0
    js_div_dynper = 0.0

    for i in range(15):
        samples_p = posterior_samples[:,0,i].numpy()
        samples_q = dynesty_posterior[:,i]
        samples_t = lrs.params[:,i,0].numpy()
        logratios = lrs.logratios[:,i].numpy()
        weights = np.exp(logratios)
        js_div_dyn += js_divergence(samples_p, samples_q)
        js_div_per += js_divergence_hist(samples_p, samples_t, weights)
        js_div_dynper += js_divergence_hist(samples_q, samples_t, weights)
    

    js_div_dyn = js_div_dyn / 15
    js_div_per = js_div_per / 15
    js_div_dynper = js_div_dynper / 15

    print(f"Jensen-Shannon Divergence with Dynesty: {js_div_dyn:.4f}")
    print(f"Jensen-Shannon Divergence with Peregrine TMNRE: {js_div_per:.4f}")
    print(f"Jensen-Shannon Divergence between Dynesty and Peregrine TMNRE: {js_div_dynper:.4f}")
