import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Sample data
x = np.array([134272, 26752, 13312, 1280, 128])  # simulation number variables
y = np.array([-4.79168, -2.19829, -0.26212, 6.31091, 8.74037])  # val loss variables

# Take log of the data
log_x = np.log(x)
log_y = np.log(y)

# Perform linear regression in log space
slope, intercept, r_value, _, _ = linregress(log_x, log_y)

# Generate best-fit line in log space
x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)  # Log-spaced x values
y_fit = np.exp(intercept) * x_fit**slope  # Convert back from log space

# Plot log-log graph
plt.figure(figsize=(8, 6))
plt.loglog(x, y, 'o', label="Data")  # Plot original data
plt.loglog(x_fit, y_fit, 'r-', label=f"Best Fit: y = {np.exp(intercept):.2f}x^{slope:.2f}")  # Best fit line

# Labels and title
plt.xlabel('Number of training simulations')
plt.ylabel('Shifted Validation Loss (L - L_{CR})')
plt.title('How Validation Loss changes with number of Training Simulations')
plt.legend()

# Show the plot
plt.show()

# Print the equation of the best fit line
print(f"Best fit equation: y = {np.exp(intercept):.2f} * x^{slope:.2f}")
