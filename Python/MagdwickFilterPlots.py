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
    "Test2_Rotating.csv"
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

t_left = left_data["recvAt"].to_numpy()


x_right = right_data["x"].to_numpy()
y_right = right_data["y"].to_numpy()
z_right = right_data["z"].to_numpy()

t_right = right_data["recvAt"].to_numpy()


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

t_left_sec = (t_left - t_left[0]) / 1000.0
t_right_sec = (t_right - t_right[0]) / 1000.0

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
# INVERT ALL GRAPHS
# ============================================================

# Magnitude
magnitude_left_raw = -magnitude_left_raw
magnitude_right_raw = -magnitude_right_raw

magnitude_left_filtered = -magnitude_left_filtered
magnitude_right_filtered = -magnitude_right_filtered

# Individual axes - raw
x_left = -x_left
y_left = -y_left
z_left = -z_left

x_right = -x_right
y_right = -y_right
z_right = -z_right

# Individual axes - filtered
earth_acceleration_left = -earth_acceleration_left
earth_acceleration_right = -earth_acceleration_right


# ============================================================
# WINDOW 1: Magnitude - LEFT and RIGHT, each raw + filtered
# ============================================================

fig1, (ax_mag_left, ax_mag_right) = plt.subplots(
    2, 1, figsize=(14, 10), sharex=True
)

ax_mag_left.plot(
    t_left_sec,
    magnitude_left_raw,
    label="Unfiltered",
    alpha=0.6
)
ax_mag_left.plot(
    t_left_sec,
    magnitude_left_filtered,
    label="Fusion filtered",
    linewidth=2
)
ax_mag_left.set_ylabel("Acceleration [m/s²]")
ax_mag_left.set_title("LEFT Sensor - Magnitude (Unfiltered vs Fusion Filtered)")
ax_mag_left.legend()
ax_mag_left.grid(True)

ax_mag_right.plot(
    t_right_sec,
    magnitude_right_raw,
    label="Unfiltered",
    alpha=0.6
)
ax_mag_right.plot(
    t_right_sec,
    magnitude_right_filtered,
    label="Fusion filtered",
    linewidth=2
)
ax_mag_right.set_xlabel("Time [s]")
ax_mag_right.set_ylabel("Acceleration [m/s²]")
ax_mag_right.set_title("RIGHT Sensor - Magnitude (Unfiltered vs Fusion Filtered)")
ax_mag_right.legend()
ax_mag_right.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# WINDOW 2: Individual axes - LEFT X/Y/Z and RIGHT X/Y/Z,
# each raw + filtered, one subplot per axis per sensor
# ============================================================

fig2, axes2 = plt.subplots(
    6, 1, figsize=(14, 18), sharex=False
)

(ax_lx, ax_ly, ax_lz, ax_rx, ax_ry, ax_rz) = axes2

# LEFT X
ax_lx.plot(t_left_sec, x_left, label="Raw X", alpha=0.5)
ax_lx.plot(t_left_sec, earth_acceleration_left[:, 0], label="Filtered X", linewidth=2)
ax_lx.set_ylabel("Accel [m/s²]")
ax_lx.set_title("LEFT Sensor - X axis")
ax_lx.legend()
ax_lx.grid(True)

# LEFT Y
ax_ly.plot(t_left_sec, y_left, label="Raw Y", alpha=0.5)
ax_ly.plot(t_left_sec, earth_acceleration_left[:, 1], label="Filtered Y", linewidth=2)
ax_ly.set_ylabel("Accel [m/s²]")
ax_ly.set_title("LEFT Sensor - Y axis")
ax_ly.legend()
ax_ly.grid(True)

# LEFT Z
ax_lz.plot(t_left_sec, z_left, label="Raw Z", alpha=0.5)
ax_lz.plot(t_left_sec, earth_acceleration_left[:, 2], label="Filtered Z", linewidth=2)
ax_lz.set_xlabel("Time [s]")
ax_lz.set_ylabel("Accel [m/s²]")
ax_lz.set_title("LEFT Sensor - Z axis")
ax_lz.legend()
ax_lz.grid(True)

# RIGHT X
ax_rx.plot(t_right_sec, x_right, label="Raw X", alpha=0.5)
ax_rx.plot(t_right_sec, earth_acceleration_right[:, 0], label="Filtered X", linewidth=2)
ax_rx.set_ylabel("Accel [m/s²]")
ax_rx.set_title("RIGHT Sensor - X axis")
ax_rx.legend()
ax_rx.grid(True)

# RIGHT Y
ax_ry.plot(t_right_sec, y_right, label="Raw Y", alpha=0.5)
ax_ry.plot(t_right_sec, earth_acceleration_right[:, 1], label="Filtered Y", linewidth=2)
ax_ry.set_ylabel("Accel [m/s²]")
ax_ry.set_title("RIGHT Sensor - Y axis")
ax_ry.legend()
ax_ry.grid(True)

# RIGHT Z
ax_rz.plot(t_right_sec, z_right, label="Raw Z", alpha=0.5)
ax_rz.plot(t_right_sec, earth_acceleration_right[:, 2], label="Filtered Z", linewidth=2)
ax_rz.set_xlabel("Time [s]")
ax_rz.set_ylabel("Accel [m/s²]")
ax_rz.set_title("RIGHT Sensor - Z axis")
ax_rz.legend()
ax_rz.grid(True)

plt.tight_layout()
plt.show()
