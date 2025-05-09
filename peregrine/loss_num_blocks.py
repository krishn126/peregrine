import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Sample data
x1 = np.array([1, 2, 3, 4, 5, 7, 10])  # num blocks 
y1 = np.array([-0.17711, -0.26212, -0.77094, -0.92246, -0.39843, -0.83181, -1.23354])  # val loss variables

x2 = np.array([1 ,3, 5, 7, 10])  # num transforms
y2 = np.array([5.24565, -0.52969, -0.26212, 0.25091, 0.40129])  # test loss variables

approx_crb = 52.545037343796665
# y = y + approx_crb  # Shift the validation loss to be positive

# Plot scatter graph
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
axs[0].scatter(x1, y1, marker='o', label="Val Loss", color = 'red')  # Plot original data
axs[0].set_xlabel("Number of Blocks")
axs[0].set_ylabel("$\mathcal{L}$")
axs[1].scatter(x2, y2, marker='x', label="Val Loss", color = 'blue')  # Plot original data
axs[1].set_xlabel("Number of Transformations")
axs[1].set_ylabel("$\mathcal{L}$")

# Labels and title

# Show the plot
plt.show()
