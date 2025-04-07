import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Sample data
x = np.array([100, 50, 20, 10, 5, 1])  # simulation number variables
x_num = np.array([14933992, 14713292, 14628872, 14608732, 14600162, 14594026])  # simulation number variables
y = np.array([-0.55154, -0.26212, -0.04276, 0.46136, 0.53392, 4.14938])  # val loss variables

approx_crb = 52.545037343796665
y = y + approx_crb  # Shift the validation loss to be positive

x_prime = x[:-1]
x_num_prime = x_num[:-1]
y_prime = y[:-1]

# Take log of the data
log_x = np.log(x)
log_x_num = np.log(x_num)
log_y = np.log(y)

log_x_prime = np.log(x_prime)
log_x_num_prime = np.log(x_num_prime)
log_y_prime = np.log(y_prime)

# Perform linear regression in log space,
slope, intercept, r_value, _, _ = linregress(log_x_num_prime, log_y_prime)

# Generate best-fit line in log space
x_fit = np.logspace(np.log10(min(x_num_prime)), np.log10(max(x_num_prime)), 100)  # Log-spaced x values
y_fit = np.exp(intercept) * x_fit**slope  # Convert back from log space

# Plot log-log graph
plt.figure(figsize=(8, 6))
plt.loglog(x_num, y, 'o', label="Val Loss")  # Plot original data
#extend line of best fit to x=1
# x_fit = np.append(x_fit, 1)
# y_fit = np.append(y_fit, np.exp(intercept) * 1**slope)
plt.loglog(x_fit, y_fit, 'r--', label=f"Best Fit: y = {np.exp(intercept):.2f}x^{slope:.3f}")  # Best fit line

# Labels and title
plt.xlabel('Number of Trainable Parameters')
plt.ylabel('Shifted Validation Loss (L - L_CR)')
plt.title('How Validation Loss Changes with Number of Trainable Parameters')
plt.legend()

# Show the plot
plt.show()

# Print the equation of the best fit line
print(f"Best fit equation: y = {np.exp(intercept):.2f} * x^{slope:.2f}")
