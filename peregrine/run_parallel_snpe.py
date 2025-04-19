from config_utils_snpe import read_config, init_config
from simulator_utils_snpe import init_simulator, simulate
from inference_utils_snpe_custom import setup_zarr_store, load_bounds
import sys
import numpy as np

if __name__ == "__main__":
    args = sys.argv[1:]
    round_id = int(args[1])
    if "coverage" not in args:
        coverage = False
    else:
        coverage = True
    snpe_parser = read_config(args)
    conf = init_config(snpe_parser, args, sim=True)
    if coverage:
        conf["zarr_params"]["chunk_size"] = 50
    if round_id == 1:
        bounds = load_bounds(conf, round_id)
        simulator = init_simulator(conf, bounds=bounds, proposal_samples=None)
    else:
        proposal_samples_path = sys.argv[2]
        proposal_samples = np.load(proposal_samples_path)
        simulator = init_simulator(conf, bounds=None, proposal_samples=proposal_samples)
    store = setup_zarr_store(conf, simulator, round_id=round_id, coverage=coverage)
    while store.sims_required > 0:
        simulate(
            simulator, store, conf, max_sims=int(conf["zarr_params"]["chunk_size"])
        )
