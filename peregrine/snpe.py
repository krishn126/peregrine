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
from inference_utils_snpe import (
    setup_zarr_store,
    setup_dataloader,
    setup_density_estimator,
    init_network,
    save_bounds,
    load_bounds,
)

# For parallelisation
import subprocess
import psutil
import logging

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
                        "run_parallel.py",
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
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Setting up trainer for round {round_id}"
        )
        trainer = setup_trainer(trainer_dir, conf, round_id)

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Initialising network for round {round_id}"
        )
        network = init_network(conf)

        if (
            not conf["snpe"]["infer_only"]
            or len(glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")) == 0
        ):
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Training network for round {round_id}"
            )
            trainer.fit(network, train_data, val_data)
            logging.info(
                f"Training completed for round {round_id}, checkpoint available at {glob.glob(f'{trainer_dir}/epoch*_R{round_id}.ckpt')[0]}"
            )

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Generate prior samples"
        )
        prior_sim = init_simulator(conf, load_bounds(conf, round_id))
        prior_samples = prior_sim.sample(100_000, targets=["z_total"])

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Generate posterior samples"
        )
        trainer.test(
            network, val_data, glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")[0]
        )
        logratios = trainer.infer(
            network, obs, prior_samples.get_dataloader(batch_size=2048)
        )
        logging.info(f"Logratios saved for round {round_id}")
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Saving logratios from round {round_id}"
        )
        save_logratios(logratios, conf, round_id)
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Update bounds from round {round_id}"
        )
        bounds = (
            sl.bounds.get_recte_bounds(logratios, threshold=conf["snpe"]["bounds_th"])
            .bounds.squeeze(1)
            .numpy()
        )
        save_bounds(bounds, conf, round_id)
        end_time = datetime.now()
        logging.info(f"Completed round {round_id}")
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [snpe.py] | Completed round {round_id} in {end_time - start_time}."
        )
