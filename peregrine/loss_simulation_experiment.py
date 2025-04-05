import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Sample data
x = np.array([134272, 67072, 26752, 13312, 6656, 2560, 1280])  # simulation number variables #128
y = np.array([-4.79168, -3.5644, -2.19829, -0.26212, 0.91438, 5.24674, 6.31091])  # val loss variables #8.74037

# x = np.array([134272, 67072, 26752, 13312, 6656])  # simulation number variables
# y = np.array([-4.79168, -3.5644, -2.19829, -0.26212, 0.91438])  # val loss variables

x_1 = np.array([1792, 2688, 4480, 8960, 19968, 29952, 49920, 62976, 89984, 107904])  # simulation number variables
y_1 = np.array([-2.5372467041015625, -2.1801271438598633, -3.156132936477661, -3.2031116485595703, -3.3537256717681885, -3.4139301776885986, -4.036552429199219, -4.062026500701904,  -4.175949573516846, -4.175949573516846])  # test loss variables

approx_crb = 52.545037343796665
y = y + approx_crb  # Shift the validation loss 
y_1 = y_1 + approx_crb  # Shift the validation loss 
# Take log of the data
log_x = np.log(x)
log_y = np.log(y)

log_x_1 = np.log(x_1)
log_y_1 = np.log(y_1)
# Perform linear regression in log space
slope, intercept, r_value, _, _ = linregress(log_x, log_y)
slope_1, intercept_1, r_value_1, _, _ = linregress(log_x_1, log_y_1)

# Generate best-fit line in log space
x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)  # Log-spaced x values
y_fit = np.exp(intercept) * x_fit**slope  # Convert back from log space

# Generate best-fit line in log space
x_fit_1 = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)  # Log-spaced x values
y_fit_1 = np.exp(intercept_1) * x_fit_1**slope_1  # Convert back from log space

# Plot log-log graph
plt.figure(figsize=(8, 6))
plt.loglog(x, y, 'o', label="Val Loss SNPE")  # Plot original data
plt.loglog(x_fit, y_fit, 'r--', label=f"Best Fit SNPE: y = {np.exp(intercept):.2f}x^{slope:.3f}")  # Best fit line
plt.loglog(x_1, y_1, 'x', label="Test Loss TMNRE")  # Plot original data
plt.loglog(x_fit_1, y_fit_1, 'g--', label=f"Best Fit TMNRE: y = {np.exp(intercept_1):.2f}x^{slope_1:.3f}")  # Best fit line

# Labels and title
plt.xlabel('Number of Training Simulations')
plt.ylabel('Shifted Validation Loss (L - L_CR)')
plt.title('How Validation Loss Changes with Number of Training Simulations for SNPE and TMNRE')
plt.legend()

# Show the plot
plt.show()

# Print the equation of the best fit line
print(f"SNPE Best fit equation: y = {np.exp(intercept):.2f} * x^{slope:.3f}")
print(f"TMNRE Best fit equation: y = {np.exp(intercept_1):.2f} * x^{slope_1:.3f}")
