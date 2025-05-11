print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Load Posteriors (Single Round NPE)
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
    """
    Custom sine-distributed probability distribution over [0, pi],
    defined by p(x) = (1/2) * sin(x).
    """   
    support = dist.constraints.interval(torch.tensor([0.0]), torch.pi)  # Support is [0, pi]

    def __init__(self, validate_args=None):
        self.low = torch.tensor([0.0]) # Lower bound
        self.high = torch.pi
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
        self.low = -torch.pi / 2  # Lower bound
        self.high = torch.pi / 2
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

    lows = torch.tensor([r[0] for r in ranges])
    highs = torch.tensor([r[1] for r in ranges])
    support = dist.constraints.independent(dist.constraints.interval(lows, highs), 1)

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
        lows = torch.tensor([priors[key].low for key in self.keys_order])
        highs = torch.tensor([priors[key].high for key in self.keys_order])
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

    joint_prior = JointPriorTensor(priors, keys_order=order)
    
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

    true_params = torch.tensor([true_params])
    round_id = 7

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
    
    loaded_density_estimator = torch.load('/data/kn405/Code/peregrine/peregrine/final_round_zoomed_in_de.pt')
    loaded_posterior = torch.load('/data/kn405/Code/peregrine/peregrine/final_round_zoomed_in_posterior.pt')
    posterior_samples = loaded_posterior.sample_batched(torch.Size([10000]), x=obs)
    posterior_samples = posterior_samples.cpu()
    #save posterior samples
    np.save('/data/kn405/Code/peregrine/peregrine/posterior_samples_zoom_in_npe.npy', posterior_samples)

    # ranges = [
    #     (0.125, 1.0),  # mass_ratio
    #     (25.0, 100.0),  # chirp_mass
    #     (0.0, 3.14159),  # theta_jn
    #     (0.0, 6.28318),  # phase
    #     (0.0, 3.14159),  # tilt_1
    #     (0.0, 3.14159),  # tilt_2
    #     (0.05, 1.0),  # a_1
    #     (0.05, 1.0),  # a_2
    #     (0.0, 6.28318),  # phi_12
    #     (0.0, 6.28318),  # phi_jl
    #     (100.0, 1500.0),  # luminosity_distance
    #     (-1.57079, 1.57079),  # dec
    #     (0.0, 6.28318),  # ra
    #     (0.0, 3.14159),  # psi
    #     (-0.1, 0.1),  # geocent_time
    # ]
    ranges = [
        (3.576977252960205078e-01,9.999501705169677734e-01),  # mass_ratio
        (2.849053955078125000e+01,3.366965866088867188e+01),  # chirp_mass
        (1.082168892025947571e-02,1.287480473518371582e+00),  # theta_jn
        (5.015052738599479198e-04,6.282635688781738281e+00),  # phase
        (1.804005727171897888e-02,3.125791549682617188e+00),  # tilt_1
        (1.731673255562782288e-02,3.126497983932495117e+00),  # tilt_2
        (5.006928741931915283e-02,9.999464750289916992e-01),  # a_1
        (5.009920150041580200e-02,9.999063014984130859e-01),  # a_2
        (7.937396294437348843e-04,6.282266139984130859e+00),  # phi_12
        (7.755699157714843750e-01,4.403653621673583984e+00),  # phi_jl
        (3.640895080566406250e+02,1.145086425781250000e+03),  # luminosity_distance
        (-6.177333369851112366e-02,2.037448883056640625e-01),  # dec
        (5.483808994293212891e+00,5.640668869018554688e+00),  # ra
        (2.063730207737535238e-04,3.141472339630126953e+00),  # psi
        (-3.634473541751503944e-03,2.502904739230871201e-03),  # geocent_time
    ]
    # # Plot posterior
    lrs = pd.read_pickle(f'/data/kn405/Code/peregrine/peregrine/logratios_R{round_id}')

    def load_dynesty(folder):
        with open(glob.glob(f"{folder}/dynesty_result.json")[0], "r") as f:
            dynesty = json.load(f)
        return dynesty["posterior"]["content"]

    dynesty_posterior = load_dynesty("/data/kn405/Code/peregrine_snpe/peregrine/peregrine")
    dynesty_posterior = np.array([dynesty_posterior[key] for key in order]).T
    

    def calculate_crb(posterior_samples):
        """
        Approximate the CRB using posterior samples by calculating the inverse of the variance.

        Args:
            posterior_samples: Tensor of shape [N_samples, D_params] containing posterior samples.

        Returns:
            CRB: Cramér-Rao Bound as an approximation.
        """
        posterior_samples = posterior_samples.squeeze(1)
        # Compute the covariance matrix from posterior samples
        covariance_matrix = np.cov(posterior_samples.T)

        # Compute the CRB as the trace of the inverse of the covariance matrix
        eigvals = np.linalg.eigvalsh(covariance_matrix)
        if np.any(eigvals <= 0):
            print("Warning: covariance matrix is not positive definite.")
        log_trace = np.sum(np.log(eigvals[eigvals > 0]))  # Avoid invalid log

        return log_trace
    
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
        pdf_p /= np.trapz(pdf_p, eval_points)
        pdf_q /= np.trapz(pdf_q, eval_points)
        
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
                
        hist_p /= hist_p.sum()
        hist_q /= hist_q.sum()
        
        hist_m = 0.5 * (hist_p + hist_q)
        
        mask = (hist_p > 0) & (hist_q > 0)
        kl_p_m = entropy(hist_p[mask], hist_m[mask])
        kl_q_m = entropy(hist_q[mask], hist_m[mask])
        
        js_div = 0.5 * (kl_p_m + kl_q_m)
        return js_div
    
    #CRB Bound
    # def crb_from_posterior_samples(posterior_samples):
    #     CRB = calculate_crb(posterior_samples)
    #     print(f"Approximate Cramér-Rao Bound (CRB): {CRB}")
    log_trace = calculate_crb(posterior_samples)
    print(f"log trace of CRB approximation: {log_trace}")
    sys.exit()

    # Plot SNPE vs TMNRE posterior
    def plot_posterior_npe_tmnre():
        fig = plt.figure(figsize=(15, 8))

        for idx in range(15):
            ax = plt.subplot(5, 3, idx + 1)
            ax.set_title(f"{label_order[idx]}")
            logratios = lrs.logratios[:, idx]
            params = lrs.params[:, idx, 0]
            weights1 = np.ones_like(posterior_samples[:,0,idx])
            weights2 = np.exp(logratios.numpy())
            plt.hist([posterior_samples[:,0,idx].numpy(), params], weights = [weights1, weights2], range=ranges[idx], bins=100, density=True, alpha=0.7)
            plt.axvline(x=true_params[:,idx], linestyle='--')

        fig.suptitle("")
        plt.tight_layout()
        blue_line = mlines.Line2D([], [], color='blue', label=f'NPE')
        orange_line = mlines.Line2D([], [], color='orange', label=f'TMNRE Round {round_id}')
        fig.legend(handles=[blue_line, orange_line], loc="upper right", fontsize=10)

        return fig

    # Plot SNPE vs Priors
    def plot_posterior_vs_prior():
        fig = plt.figure(figsize=(15, 8))
        joint_prior_sample = joint_prior.sample(torch.Size([100000])).cpu()

        for idx in range(15):
            ax = plt.subplot(5, 3, idx + 1)
            ax.set_title(f"{order[idx]}")
            plt.hist([posterior_samples[:,0,idx].numpy(), joint_prior_sample[:,idx].numpy()], range=ranges[idx], bins=100, density=True, alpha=0.7)
            plt.axvline(x=true_params[:,idx], linestyle='--')

        fig.suptitle(f"SNPE Round {round_id} vs Priors", fontsize=20)
        plt.tight_layout()
        blue_line = mlines.Line2D([], [], color='blue', label='SNPE')
        orange_line = mlines.Line2D([], [], color='orange', label='Priors')
        fig.legend(handles=[blue_line, orange_line], loc="upper right", fontsize=10)

        return fig
    
    # #Corner Plot of SNPE vs Dynesty
    def corner_plot(): 
        fig = plt.figure(figsize=(15, 8))       

        fig = corner.corner(posterior_samples[:,0,:].numpy(), color='blue', range=ranges, labels=label_order, hist_kwargs={"density": True})
        corner.corner(dynesty_posterior, color='red', fig=fig, range=ranges, hist_kwargs={"density": True})
        blue_line = mlines.Line2D([], [], color='blue', label=f'NPE')
        red_line = mlines.Line2D([], [], color='red', label='Dynesty')
        fig.legend(handles=[blue_line, red_line], loc="upper right", fontsize=40) # Add the legend

        return fig
    
    # #Save Plots
    fig = plot_posterior_npe_tmnre  ()
    plt.savefig(f"/data/kn405/Code/peregrine/posterior_plots/zoomed_in_npe_vs_tmnre.png", dpi=300, bbox_inches='tight')

    #JS Divergence Calculations 
    def js_div_calcs():
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
        print(f"Jensen-Shannon Divergence with Peregrine TMNRE Round {round_id}: {js_div_per:.4f}")
        print(f"Jensen-Shannon Divergence between Dynesty and Peregrine TMNRE Round {round_id}: {js_div_dynper:.4f}")
    
    js_div_calcs()
    
