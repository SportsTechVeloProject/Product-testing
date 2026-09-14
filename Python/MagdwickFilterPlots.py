import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import imufusion


# ============================================================
# Read CSV
# ============================================================

file_path = os.path.join(
    os.path.dirname(__file__),
    "Recording1_IMU.csv"
)

data = pd.read_csv(file_path)


# ============================================================
# Separate left and right sensors
# ============================================================

left_data = data[data["sensor"] == "left"].copy()
right_data = data[data["sensor"] == "right"].copy()

left_data = left_data.reset_index(drop=True)
right_data = right_data.reset_index(drop=True)


# ============================================================
# ACCELEROMETER
# ============================================================

x_left = left_data["x"].to_numpy()
y_left = left_data["y"].to_numpy()
z_left = left_data["z"].to_numpy()

t_left = left_data["t"].to_numpy()


x_right = right_data["x"].to_numpy()
y_right = right_data["y"].to_numpy()
z_right = right_data["z"].to_numpy()

t_right = right_data["t"].to_numpy()


# ============================================================
# Remove gravity from Y-axis?
# ============================================================

#y_left = y_left - 9.82
#y_right = y_right - 9.82


# ============================================================
# Raw acceleration magnitude
# ============================================================

magnitude_left_raw = np.sqrt(
    x_left**2 +
    y_left**2 +
    z_left**2
)

magnitude_right_raw = np.sqrt(
    x_right**2 +
    y_right**2 +
    z_right**2
)


# ============================================================
# Time
# ============================================================

t_left_sec = (t_left - t_left[0])
t_right_sec = (t_right - t_right[0])

#============================================================
#Data to be used in the fusion filter
#============================================================

gx_left = left_data["gx"].to_numpy()
gy_left = left_data["gy"].to_numpy()
gz_left = left_data["gz"].to_numpy()

gx_right = right_data["gx"].to_numpy()
gy_right = right_data["gy"].to_numpy()
gz_right = right_data["gz"].to_numpy()


# ============================================================
# Sample rate
# ============================================================

sample_rate = 100.0  # Hz

dt_left = 1.0 / sample_rate
dt_right = 1.0 / sample_rate


# ============================================================
# Create Fusion filters
# ============================================================
# IMPORTANT:
# Use a separate AHRS object for each sensor.

ahrs_left = imufusion.Ahrs()
ahrs_right = imufusion.Ahrs()

# ============================================================
# Fusion settings
# ============================================================

settings_left = imufusion.AhrsSettings()

settings_left.convention = imufusion.CONVENTION_NWU
settings_left.gain = 0.5
settings_left.acceleration_rejection = 10
settings_left.magnetic_rejection = 10
settings_left.rejection_timeout = 5

ahrs_left.set_settings(settings_left)

settings_right = imufusion.AhrsSettings()

settings_right.convention = imufusion.CONVENTION_NWU
settings_right.gain = 0.5
settings_right.acceleration_rejection = 10
settings_right.magnetic_rejection = 10
settings_right.rejection_timeout = 5

ahrs_right.set_settings(settings_right)


# ============================================================
# Store Fusion results
# ============================================================

earth_acceleration_left = []
earth_acceleration_right = []


# ============================================================
# Run Fusion - LEFT
# ============================================================

for i in range(len(left_data)):

    accel = np.array([
        x_left[i],
        y_left[i],
        z_left[i]
    ])

    gyro = np.array([
        gx_left[i],
        gy_left[i],
        gz_left[i]
    ])

    ahrs_left.update_no_magnetometer(
        gyro,
        accel,
        #dt_left
    )

    earth_acceleration_left.append(
        ahrs_left.get_earth_acceleration()  
    )


# ============================================================
# Run Fusion - RIGHT
# ============================================================

for i in range(len(right_data)):

    accel = np.array([
        x_right[i],
        y_right[i],
        z_right[i]
    ])

    gyro = np.array([
        gx_right[i],
        gy_right[i],
        gz_right[i]
    ])

    ahrs_right.update_no_magnetometer(
        gyro,
        accel,
        #dt_right
    )

    earth_acceleration_right.append(
        ahrs_right.get_earth_acceleration()
    )


# ============================================================
# Convert Fusion results to NumPy arrays
# ============================================================

earth_acceleration_left = np.array(
    earth_acceleration_left
)

earth_acceleration_right = np.array(
    earth_acceleration_right
)


# ============================================================
# Calculate filtered acceleration magnitude
# ============================================================

magnitude_left_filtered = np.sqrt(
    earth_acceleration_left[:, 0]**2 +
    earth_acceleration_left[:, 1]**2 +
    earth_acceleration_left[:, 2]**2
)

magnitude_right_filtered = np.sqrt(
    earth_acceleration_right[:, 0]**2 +
    earth_acceleration_right[:, 1]**2 +
    earth_acceleration_right[:, 2]**2
)


# ============================================================
# PLOT LEFT SENSOR
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    t_left_sec,
    magnitude_left_raw,
    label="Unfiltered",
    alpha=0.6
)

plt.plot(
    t_left_sec,
    magnitude_left_filtered,
    label="Fusion filtered",
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("LEFT Sensor - Unfiltered vs Fusion Filtered")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ============================================================
# PLOT RIGHT SENSOR
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    t_right_sec,
    magnitude_right_raw,
    label="Unfiltered",
    alpha=0.6
)

plt.plot(
    t_right_sec,
    magnitude_right_filtered,
    label="Fusion filtered",
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("RIGHT Sensor - Unfiltered vs Fusion Filtered")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ============================================================
# Plot individual axes - RIGHT sensor
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    t_right_sec,
    x_right,
    label="Raw X",
    alpha=0.5
)

plt.plot(
    t_right_sec,
    earth_acceleration_right[:, 0],
    label="Filtered X",
    linewidth=2
)

plt.plot(
    t_right_sec,
    y_right,
    label="Raw Y",
    alpha=0.5
)

plt.plot(
    t_right_sec,
    earth_acceleration_right[:, 1],
    label="Filtered Y",
    linewidth=2
)

plt.plot(
    t_right_sec,
    z_right,
    label="Raw Z",
    alpha=0.5
)

plt.plot(
    t_right_sec,
    earth_acceleration_right[:, 2],
    label="Filtered Z",
    linewidth=2
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("RIGHT Sensor - Individual Axes")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
# ============================================================
# Save filtered acceleration magnitude
# ============================================================

#save_right_data = pd.DataFrame({"t": t_right_sec, "acc": magnitude_right_filtered})
#save_right_data.to_csv("Python/madgwick_right_data.csv", index=False)