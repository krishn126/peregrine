print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: SNPE Inference
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
from datetime import datetime
import glob
import pickle
import swyft.lightning as sl
from config_utils_snpe import read_config, init_config
from simulator_utils_snpe import init_simulator, simulate
from inference_utils_snpe_custom import (
    init_network,
    setup_zarr_store,
    setup_dataloader,
    setup_density_estimator,
    save_bounds,
    load_bounds,
)
from sbi.inference import SNPE
import torch
import torch.distributions as dist
from sbi.utils import BoxUniform
from sbi.inference.posteriors import DirectPosterior
import torch.nn.functional as F

# For parallelisation
import subprocess
import psutil
import logging

import matplotlib.pyplot as plt
import os
import tqdm
import wandb
from torch.optim import AdamW

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
    wandb.init(project="npe4gw_custom_loops") # initialise wandb
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

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Setting up trainer and network for round {round_id}"
        )

        # # Define priors 
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

        # Create a joint prior distribution
        joint_prior = JointPriorTensor(priors, keys_order=order)
        dummy_theta = joint_prior.sample(torch.Size([64]))  # Sample 64 thetas
        dummy_x = torch.randn(64, 6, 49152)
        # Define the inference object
        density_estimator = setup_density_estimator(conf, dummy_theta, dummy_x)
        
        if (
            not conf["snpe"]["infer_only"]
            or len(glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")) == 0
        ):
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Training network for round {round_id}"
            )
            
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
            
            def setup_scheduler(optimizer):
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=10,
                    verbose=True,
                    threshold=1e-4,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=0,
                    eps=1e-8,
                )
                return scheduler
            
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

            # num_epochs = conf["hyperparams"]["num_epochs"]
            num_epochs = 20
            optimizer = AdamW(density_estimator.parameters(), lr=1e-3) # initialise pytorch optimiser
            scheduler = setup_scheduler(optimizer) # initialise scheduler
            step = 0
            epoch_val_loss = 0.0

            num_train_batches = sum(1 for _ in train_data)
            num_val_batches = sum(1 for _ in val_data)

            # Train the density estimator
            for epoch in range(num_epochs):
                density_estimator.train() # put estimator into train mode
                train_loss_epoch = 0.0
                # num_batches = len(train_data)
                with tqdm.tqdm(
                    train_data, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
                ) as pbar: # Fancy tqdm loading bar for printing the training status
                        #iterate through the training examples
                    for sample in pbar:
                        theta_train = get_theta(sample)
                        x_train = get_data(sample)
                        loss = density_estimator.loss(theta_train, x_train).mean() # compute loss on batch
                        optimizer.zero_grad() # zero the optimiser
                        loss.backward() # compute the gradients
                        optimizer.step() # take a step given these gradients
                        # train_losses.append(loss.item()) # track losses
                        train_loss_epoch += loss.item()
                        wandb.log({"train_loss": loss.item()}) # log loss to wandb
                        step += 1
                        pbar.set_postfix(
                            {
                                "Train Loss": f"{loss.item():.4f}| Val Loss: {epoch_val_loss:.4f}"
                            }
                        ) # print to tqdm bar
                avg_train_loss = train_loss_epoch / num_train_batches

                density_estimator.eval() # put estimator into eval mode

                epoch_val_loss = 0.0
                with torch.no_grad(): # ensure no gradients computed in val mode
                    for sample in val_data: 
                        theta_val = get_theta(sample)
                        x_val = get_data(sample)   # iterate through validation dataloader
                        epoch_val_loss += density_estimator.loss(theta_val, x_val).mean().item() # compute overall loss on val dataset batch by batch

                epoch_val_loss /= num_val_batches # average loss over val dataset        
                scheduler.step(epoch_val_loss)
                learning_rate = scheduler.get_last_lr()
                wandb.log(
                    {
                        "val_loss": epoch_val_loss,
                        "step": step,
                        "learning_rate": learning_rate,
                    }
                ) # log results

            posterior = DirectPosterior(density_estimator, joint_prior)
            # Plot posterior
            posterior_samples = posterior.sample_batched(torch.Size([5000]), x=obs)   
            for i in range(15):
                plt.figure(figsize=(8, 5))
                plt.hist(posterior_samples[:,0,i].numpy(), bins=30, density=True, alpha=0.7, label="Posterior samples")
                plt.axvline(x=true_params[i], linestyle='--', label="True parameter value")
                plt.xlabel("Theta")
                plt.ylabel("Density")
                plt.title(f"Posterior Distribution for the parameter {order[i]}")
                plt.legend()
                plt.savefig(f"/data/kn405/Code/peregrine_snpe/peregrine/posterior_plots/posterior_for_{order[i]}.png", dpi=300, bbox_inches='tight')        