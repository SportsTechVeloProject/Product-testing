import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Hello World")

# Read the original CSV file
file_path = os.path.join(os.path.dirname(__file__), "squats.csv")

data = pd.read_csv(file_path)

# --------------------------------
# Separate left and right sensors
# --------------------------------

left_data = data[data["sensor"] == "left"]
right_data = data[data["sensor"] == "right"]


# --------------------------------
# Get acceleration values from the left sensor
# --------------------------------

x_left = left_data["x"].to_numpy()
y_left = left_data["y"].to_numpy()
z_left = left_data["z"].to_numpy()

t_left = left_data["recvAt"].to_numpy()


# --------------------------------
# Get acceleration values from the right sensor
# --------------------------------

x_right = right_data["x"].to_numpy()
y_right = right_data["y"].to_numpy()
z_right = right_data["z"].to_numpy()

t_right = right_data["recvAt"].to_numpy()

# --------------------------------
# Calculate acceleration magnitude
# --------------------------------

magnitude_left = np.sqrt(x_left**2 + y_left**2 + z_left**2)
magnitude_right = np.sqrt(x_right**2 + y_right**2 + z_right**2)

# --------------------------------
# Convert time to seconds (relative to start) for nicer x-axes
# --------------------------------

t_left_sec = (t_left - t_left[0]) / 1000.0
t_right_sec = (t_right - t_right[0]) / 1000.0


# ================================
# Visualize acceleration data (x, y, z, magnitude) for both sensors
# ================================

fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="col")

# --- Left sensor plots ---
axes[0, 0].plot(t_left_sec, x_left, color="tab:blue")
axes[0, 0].set_title("Left sensor - X")
axes[0, 0].set_ylabel("Acceleration")

axes[1, 0].plot(t_left_sec, y_left, color="tab:orange")
axes[1, 0].set_title("Left sensor - Y")
axes[1, 0].set_ylabel("Acceleration")

axes[2, 0].plot(t_left_sec, z_left, color="tab:green")
axes[2, 0].set_title("Left sensor - Z")
axes[2, 0].set_ylabel("Acceleration")

axes[3, 0].plot(t_left_sec, magnitude_left, color="tab:red")
axes[3, 0].set_title("Left sensor - Magnitude")
axes[3, 0].set_ylabel("Acceleration")
axes[3, 0].set_xlabel("Time (s)")

# --- Right sensor plots ---
axes[0, 1].plot(t_right_sec, x_right, color="tab:blue")
axes[0, 1].set_title("Right sensor - X")

axes[1, 1].plot(t_right_sec, y_right, color="tab:orange")
axes[1, 1].set_title("Right sensor - Y")

axes[2, 1].plot(t_right_sec, z_right, color="tab:green")
axes[2, 1].set_title("Right sensor - Z")

axes[3, 1].plot(t_right_sec, magnitude_right, color="tab:red")
axes[3, 1].set_title("Right sensor - Magnitude")
axes[3, 1].set_xlabel("Time (s)")

for ax_row in axes:
    for ax in ax_row:
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("overheadPress.png", dpi=150)
plt.show()

print("Plot saved to 'squats_acceleration_plot.png'")


# ================================
# Optional: overlay all 3 axes on a single plot per sensor
# (easier to see how the axes relate to each other in time)
# ================================

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes2[0].plot(t_left_sec, x_left, label="x", alpha=0.8)
axes2[0].plot(t_left_sec, y_left, label="y", alpha=0.8)
axes2[0].plot(t_left_sec, z_left, label="z", alpha=0.8)
axes2[0].plot(t_left_sec, magnitude_left, label="magnitude", color="black", linewidth=1.5)
axes2[0].set_title("Left sensor - all axes overlaid")
axes2[0].set_ylabel("Acceleration")
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(t_right_sec, x_right, label="x", alpha=0.8)
axes2[1].plot(t_right_sec, y_right, label="y", alpha=0.8)
axes2[1].plot(t_right_sec, z_right, label="z", alpha=0.8)
axes2[1].plot(t_right_sec, magnitude_right, label="magnitude", color="black", linewidth=1.5)
axes2[1].set_title("Right sensor - all axes overlaid")
axes2[1].set_ylabel("Acceleration")
axes2[1].set_xlabel("Time (s)")
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
