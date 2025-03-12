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

# For parallelisation
import subprocess
import logging

import matplotlib.pyplot as plt
import wandb


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
    
    loaded_posterior = torch.load('/data/kn405/Code/peregrine_snpe/peregrine/peregrine/posterior.pt')
    posterior_samples = loaded_posterior.sample_batched(torch.Size([5000]), x=obs)
    # Plot posterior
    for i in range(15):
        plt.figure(figsize=(8, 5))
        plt.hist(posterior_samples[:,0,i].numpy(), bins=30, density=True, alpha=0.7, label="Posterior samples")
        plt.axvline(x=true_params[i], linestyle='--', label="True parameter value")
        plt.xlabel(f"{order[i]}")
        plt.ylabel("Density")
        plt.title(f"Posterior Distribution for the parameter {order[i]}")
        plt.legend()
        plt.savefig(f"/data/kn405/Code/peregrine_snpe/peregrine/posterior_plots/posterior_for_{order[i]}.png", dpi=300, bbox_inches='tight')