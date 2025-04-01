print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Coverage Test
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
import numpy as np
from datetime import datetime
import glob
from config_utils_snpe import read_config, init_config
from simulator_utils_snpe import init_simulator
from inference_utils_snpe import (
    save_coverage,
    setup_zarr_store,
    setup_dataloader,
    setup_trainer,
    init_network,
    load_bounds,
)
import subprocess
import psutil
import time
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    args = sys.argv[1:]
    n_samples = int(args[1])
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Running coverage tests on {n_samples} samples per round"
    )
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Reading config file"
    )
    # Load and parse config file
    snpe_parser = read_config(args)
    conf = init_config(snpe_parser, args)
    conf["snpe"]["shuffling"] = False
    round_id = int(conf["snpe"]["num_rounds"])
    bounds = load_bounds(conf, round_id)
    simulator = init_simulator(conf, bounds)
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Initialising coverage zarrstore for round {round_id}"
    )
    coverage_store = setup_zarr_store(
        conf, simulator, round_id=round_id, coverage=True, n_sims=n_samples
    )
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Simulating coverage observations for round {round_id}"
    )
    if conf["zarr_params"]["njobs"] == -1:
        njobs = psutil.cpu_count(logical=True)
    elif conf["zarr_params"]["njobs"] > psutil.cpu_count(logical=False):
        njobs = psutil.cpu_count(logical=True)
    elif conf["zarr_params"]["run_parallel"]:
        njobs = conf["zarr_params"]["njobs"]
    else:
        njobs = 1
    while coverage_store.sims_required > 0:
        processes = []
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | (Re)starting {njobs} simulation batches. {coverage_store.sims_required} simulations still required"
        )
        for job in range(njobs):
            p = subprocess.Popen(
                [
                    "python",
                    "run_parallel.py",
                    f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.txt",
                    str(round_id),
                    f"coverage",
                ]
            )
            processes.append(p)
        status_array = np.array([None for p in processes])
        while np.all(status_array == None) and len(status_array) != 0:
            time.sleep(60)
            status_array = np.array([p.poll() for p in processes])
        for p in processes:
            p.kill()

    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [coverage.py] | Setting up dataloaders for round {round_id}"
    )
    train_data, val_data, trainer_dir = setup_dataloader(
        coverage_store, simulator, conf, round_id
    )

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

    limit = 1

    for i, sample in enumerate(train_data):
        if i > limit:
            break
        theta_train = get_theta(sample).to('cuda')
        x_train = get_data(sample).to('cuda') 
    
    print(f"theta_train: {theta_train.shape}")
    print(f"x_train: {x_train.shape}")
    sys.exit()
    
    def compute_coverage_per_param(num_sims, true_parameters, credibility_levels=torch.linspace(0.05, 0.95, 12)):
        _, num_parameters = true_parameters.shape
        num_levels = credibility_levels.shape[0]
        empirical_coverages = torch.zeros(num_parameters, num_levels)

        # Sort posterior samples along the sample axis
        for j in range(num_sims):
            posterior_samples = loaded_posterior.sample_batched(torch.Size([1000]), x=obs)
            posterior_samples = posterior_samples.squeeze(1)
            sorted_samples, _ = torch.sort(posterior_samples, dim=0)
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
            ax.set_title(f"{order[i]}")
            plt.plot(credibility_levels, empirical_coverages[i, :], marker='o', linestyle='-', alpha=0.7, label=f"Param {order[i]}")
            plt.plot([0, 1], [0, 1], 'k--', label="Ideal Calibration") # Reference y=x line for perfect calibration
            
        
        plt.xlabel("Expected Coverage")
        plt.ylabel("Empirical Coverage")
        plt.suptitle("Coverage Test for Each Parameter")
        plt.legend(loc="lower right", fontsize=8, ncol=2)  # Compact legend
        plt.grid(True)
        plt.savefig(f"/data/kn405/Code/peregrine/posterior_plots/empirical_coverage_tests.png", dpi=300, bbox_inches='tight')

    # Compute and plot empirical coverage
 
    true_params = true_params.repeat(1000, 1)
    credibility_levels, empirical_coverages = compute_coverage_per_param(1000, true_params)
    plot_coverage(credibility_levels, empirical_coverages)
