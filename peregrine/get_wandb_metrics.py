import wandb
import matplotlib.pyplot as plt
import pandas as pd
import sys
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter

api = wandb.Api()

# Load run
run1 = api.run("knanavati126-university-of-cambridge/npe4gw_custom_loops/wo8f4nnk")
run2 = api.run("knanavati126-university-of-cambridge/npe4gw_custom_loops/ayekdb0m")
run3 = api.run("knanavati126-university-of-cambridge/npe4gw_custom_loops/e2cftkc9")

rows_1 = []
for row in run1.scan_history():
    rows_1.append(row)

rows_2 = []
for row in run2.scan_history():
    rows_2.append(row)

rows_3 = []
for row in run3.scan_history():
    rows_3.append(row)

history_1 = pd.DataFrame(rows_1)
history_2 = pd.DataFrame(rows_2)
history_3 = pd.DataFrame(rows_3)

# Fetch logged metrics
valid_1 = history_1.dropna(subset=["val_loss"])
valid_2 = history_2.dropna(subset=["val_loss"])
valid_3 = history_3.dropna(subset=["val_loss_round_1"])
valid_4 = history_3.dropna(subset=["val_loss_round_2"])
valid_5 = history_3.dropna(subset=["val_loss_round_3"])

# Example: Plotting training loss vs. steps
plt.plot(history_1["_step"], history_1["train_loss"], label="Train Loss - MAF", color="lime")
plt.plot(valid_1["_step"], valid_1["val_loss"], label="Validation Loss - MAF", color="blue", linestyle="--")
plt.plot(history_2["_step"], history_2["train_loss"], label="Train Loss - NSF", color="pink")
plt.plot(valid_2["_step"], valid_2["val_loss"], label="Validation Loss - NSF", color="red", linestyle="--")
plt.xlabel("Step")
plt.ylabel("$\mathcal{L}$")
plt.legend()
plt.show()

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
axs[0].plot(valid_3["epoch_round_1"], valid_3["val_loss_round_1"], label="Validation Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("$\mathcal{L}$")
axs[0].set_title("Round 1")
axs[0].legend()
axs[1].plot(valid_4["epoch_round_2"], valid_4["val_loss_round_2"], label="Validation Loss")
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # Always use scientific notation
axs[1].yaxis.set_major_formatter(formatter)
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("$\mathcal{L}$")
axs[1].set_title("Round 2")
axs[2].plot(valid_5["epoch_round_3"], valid_5["val_loss_round_3"], label="Validation Loss")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("$\mathcal{L}$")
axs[2].set_title("Round 3")
plt.tight_layout()
plt.show()
