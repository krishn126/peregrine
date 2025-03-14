import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
import corner


def load_dynesty(folder):
    with open(glob.glob(f"{folder}/dynesty_result.json")[0], "r") as f:
        dynesty = json.load(f)
    return dynesty["posterior"]["content"]

dynesty_posterior = load_dynesty("/Users/knana/Documents/GitHub/peregrine_fork/peregrine")

param_names = [
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
#plot histogram of all parameters
# plt.figure(figsize=(15, 8))
# for idx in range(len(param_names)):
#     ax = plt.subplot(5, 3, idx + 1)
#     plt.hist(
#         dynesty_posterior[param_names[idx]],
#         bins=20,
#         density=True,
#         color="blue",
#         alpha=0.5,
#     )
# plt.show()

#create dynesty_posterior object with only keys that are in param_names
dynesty_posterior = {key: dynesty_posterior[key] for key in param_names}
#create a single tensor with all 15 parameters 
dynesty_posterior = np.array([dynesty_posterior[key] for key in param_names]).T
#ensure figure is right size
fig = corner.corner(dynesty_posterior, scale=0.7)
plt.show()
