print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Multi Round SNPE Inference 
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey, K.Nanavati
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
    setup_zarr_store,
    setup_dataloader,
    setup_density_estimator,
    setup_scheduler,
    init_network,
)
import torch
import torch.distributions as dist
from sbi.inference.posteriors import DirectPosterior
import torch.nn.functional as F

# For parallelisation
import subprocess
import psutil
import logging

import tqdm
import wandb
from torch.optim import AdamW
import numpy as np
    
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
    posteriors = [] 

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

    dummy_theta = joint_prior.sample(torch.Size([64])).to('cuda')  
    dummy_x = torch.randn(64, 6, 49152).to('cuda')  

    proposal = joint_prior

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
    
    def gaussian_kernel(x, x_o, tau):
        """
        x: Tensor of shape [batch_size, 6, 49152]
        x_o: Tensor of shape [6, 49152]
        tau: float
        Returns: Tensor of shape [batch_size]
        """
        diff = x - x_o  # [batch_size, 6, 49152]
        dist_sq = torch.sum(diff ** 2, dim=(1, 2))  # [batch_size]
        return torch.exp(-dist_sq / (2 * tau ** 2))  # drop normalizer for stability
    
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

    prior_samples = joint_prior.sample(
        torch.Size([10000])
    )    
    prior_samples = prior_samples.squeeze(1)
    prior_samples = prior_samples.cpu().numpy()
    np.save(f"proposal_samples_round_1.npy", prior_samples)

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
            if round_id == 1:
                print("Using Prior Samples for Simulations")
                simulator = init_simulator(conf)
                simulate(simulator, store, conf)
            else:
                print("Using Proposal Samples for Simulations")
                simulator = init_simulator(conf, proposal_samples=proposal_samples) 
                simulate(simulator, store, conf)
        logging.info(f"Simulations for round {round_id} completed")

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Setting up dataloaders for round {round_id}"
        )
        train_data, val_data, trainer_dir = setup_dataloader(
            store, simulator, conf, round_id
        )

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Setting up trainer and network for round {round_id}"
        )

        density_estimator = setup_density_estimator(conf, dummy_theta, dummy_x)
        density_estimator = density_estimator.to('cuda')

        num_epochs = conf["hparams"]["max_epochs"]
        optimizer = AdamW(density_estimator.parameters(), lr=1e-3) 
        scheduler = setup_scheduler(optimizer) 
        step = 0
        epoch_val_loss = 0.0         

        num_train_batches = sum(1 for _ in train_data)
        num_val_batches = sum(1 for _ in val_data)
        best_validation_loss = float("inf")
        no_improvement_count = 0
        patience = 15 

        limit = int(num_train_batches)

        if (
            not conf["snpe"]["infer_only"]
            or len(glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")) == 0
        ):
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Training network for round {round_id}"
            )

            for epoch in range(num_epochs):
                density_estimator.train() 
                with tqdm.tqdm(
                    total = int(num_train_batches), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
                ) as pbar: 
                    for i, sample in enumerate(train_data):
                        if i > limit:
                            break
                        theta_train = get_theta(sample).to('cuda')                        
                        x_train = get_data(sample).to('cuda')
                        losses = density_estimator.loss(theta_train, x_train) 
                        if round_id == 1:
                            log_weights = torch.zeros_like(losses)
                            weights = torch.exp(log_weights)
                        else: 
                            with torch.no_grad():
                                log_p_theta = joint_prior.log_prob(theta_train)
                                log_q_theta = proposal.log_prob(theta_train)
                                log_weights = log_p_theta - log_q_theta
                                weights = torch.exp(log_weights)
                        loss = (weights * losses).mean()
                        optimizer.zero_grad() 
                        loss.backward() 
                        optimizer.step() 
                        wandb.log({f"train_loss_round_{round_id}": loss.item()}) 
                        step += 1
                        pbar.update(1) 
                        pbar.set_postfix(
                            {
                                "Train Loss": f"{loss.item():.4f}| Val Loss: {epoch_val_loss:.4f}| Round: {round_id}"
                            }
                        ) 

                density_estimator.eval() 
                epoch_val_loss = 0.0

                with torch.no_grad():
                    with tqdm.tqdm(
                        total = int(num_val_batches), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
                    ) as pbar: 
                        for sample in val_data:
                            theta_val = get_theta(sample).to('cuda')                        
                            x_val = get_data(sample).to('cuda')
                            losses = density_estimator.loss(theta_val, x_val) 
                            if round_id == 1:
                                log_weights = torch.zeros_like(losses)
                                weights = torch.exp(log_weights)
                            else: 
                                with torch.no_grad():
                                    log_p_theta = joint_prior.log_prob(theta_val)
                                    log_q_theta = proposal.log_prob(theta_val)
                                    log_weights = log_p_theta - log_q_theta
                                    weights = torch.exp(log_weights)
                            val_loss = (weights * losses).mean()
                            epoch_val_loss += val_loss
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "Calculating Val Loss for Epoch": f"{epoch + 1}| Round: {round_id}"
                                }
                            ) 

                epoch_val_loss /= num_val_batches       
                scheduler.step(epoch_val_loss) 
                learning_rate = optimizer.param_groups[0]["lr"]

                wandb.log(
                    {
                        f"val_loss_round_{round_id}": epoch_val_loss,
                        f"step_round_{round_id}": step,
                        f"learning_rate_round_{round_id}": learning_rate,
                        f"epoch_round_{round_id}": epoch,
                    }
                ) 

                if epoch_val_loss < best_validation_loss:
                    best_validation_loss = epoch_val_loss
                    no_improvement_count = 0  
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("Early stopping triggered.")
                        break  
                
            density_estimator.eval()
            posterior = DirectPosterior(density_estimator, joint_prior)
            posteriors.append(posterior)

            proposal = posterior.set_default_x(obs) 

            if round_id != int(conf["snpe"]["num_rounds"]):
                num_sims_proposal = int(conf["zarr_params"]["sim_schedule"][round_id])

                proposal_samples = posterior.sample_batched(
                    torch.Size([num_sims_proposal]), x=obs
                )
                proposal_samples = proposal_samples.squeeze(1)
                proposal_samples = proposal_samples.cpu().numpy()
                np.save(f"proposal_samples_round_{round_id+1}.npy", proposal_samples)
            
            print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Training for round {round_id} completed. Saving Density Estimator and Posterior"
            )
            torch.save(density_estimator, f"/data/kn405/Code/peregrine_snpe/peregrine/peregrine/round_{round_id}_density_estimator_snpe.pt")
            torch.save(posterior, f"/data/kn405/Code/peregrine_snpe/peregrine/peregrine/round_{round_id}_posterior_snpe.pt")
            