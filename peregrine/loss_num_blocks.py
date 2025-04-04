import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Sample data
x = np.array([1, 2, 3, 4, 5, 7, 10])  # simulation number variables
y = np.array([-0.17711, -0.26212, -0.77094, -0.92246, -0.39843, -0.83181, -1.23354])  # val loss variables

approx_crb = 52.545037343796665
y = y + approx_crb  # Shift the validation loss to be positive

# Plot scatter graph
plt.figure(figsize=(8, 6))
plt.scatter(x, y, marker='o', label="Val Loss")  # Plot original data

# Labels and title
plt.xlabel('Number of Blocks in Neural Spine Flow')
plt.ylabel('Shifted Validation Loss (L - L_CR)')
plt.title('How Validation Loss Changes with Number of Blocks')
plt.legend()

# Show the plot
plt.show()
