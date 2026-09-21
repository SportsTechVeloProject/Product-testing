import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import MagdwickFilter
import ValidatePeaks
import PeakDetection
import VelocityCalculations

# ============================================================
# Read CSV
# ============================================================

file_path = os.path.join(
    os.path.dirname(__file__),
    "Test1_Rotating.csv"
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
# Filter using magdwick filter 
# ============================================================

earth_acceleration_left, magnitude_left_filtered = MagdwickFilter.madgwick_filter(
    x_left,
    y_left,
    z_left,
    gx_left,
    gy_left,
    gz_left,
    dt_left
)
earth_acceleration_right, magnitude_right_filtered = MagdwickFilter.madgwick_filter(
    x_right,
    y_right,
    z_right,
    gx_right,
    gy_right,
    gz_right,
    dt_right
)

#============================================================
# INVERT ALL Values 
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


##Calcualate number of peaks
peaks_right, properties_right, treshold_right = PeakDetection.peak_detection(
    magnitude_right_filtered,
    window_size=50,  k=2,  
    distance=100    
)
peaks_left, properties_left, treshold_left = PeakDetection.peak_detection(
    magnitude_left_filtered, 
    window_size=50, k=2, 
    distance = 100)


if(peaks_left.size != peaks_right.size):
    print("Warning: The number of peaks detected in left and right data is not equal.")
    print(f"Number of peaks in left data: {peaks_left.size}")
    print(f"Number of peaks in right data: {peaks_right.size}")
else:
    print(f"Number of peaks detected in both left and right data: {peaks_left.size}")

peak_times_left = t_left_sec[peaks_left]
peak_times_right = t_right_sec[peaks_right]



# Filter false peaks
valid_peaks_right = ValidatePeaks.validate_peaks(
    magnitude_right_filtered,
    peaks_right,
    sample_rate
)

valid_peaks_left = ValidatePeaks.validate_peaks(
    magnitude_left_filtered,
    peaks_left,
    sample_rate
)


print("Peaks before filtering:", len(peaks_right))
print("Peaks after filtering:", len(valid_peaks_right))


# ============================================================
# Calculate acceleration and velocity for EVERY repetition
# ============================================================

WINDOW_SIZE = 15

results_right = VelocityCalculations.calculate_velocity(
    magnitude_right_filtered,
    t_right_sec,
    valid_peaks_right,
    window_size=WINDOW_SIZE
)

results_left = VelocityCalculations.calculate_velocity(
    magnitude_left_filtered, 
    t_left_sec, 
    valid_peaks_left,
    window_size=WINDOW_SIZE
)

# ============================================================
# Plot acceleration and detected peaks for LEFT and RIGHT sensors
# ============================================================

fig, (ax_right, ax_left) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# ----------------------------------------------------
# RIGHT sensor
# ----------------------------------------------------

ax_right.plot(
    t_right_sec,
    magnitude_right_filtered,
    label="Filtered acceleration"
)

ax_right.plot(
    t_right_sec[valid_peaks_right],
    magnitude_right_filtered[valid_peaks_right],
    "x",
    markersize=10,
    label="Detected peaks"
)

ax_right.set_ylabel("Acceleration [m/s²]")
ax_right.set_title("RIGHT Sensor - Acceleration with Detected Peaks")
ax_right.legend()
ax_right.grid(True)

# ----------------------------------------------------
# LEFT sensor
# ----------------------------------------------------

ax_left.plot(
    t_left_sec,
    magnitude_left_filtered,
    label="Filtered acceleration"
)

ax_left.plot(
    t_left_sec[valid_peaks_left],
    magnitude_left_filtered[valid_peaks_left],
    "x",
    markersize=10,
    label="Detected peaks"
)

ax_left.set_xlabel("Time [s]")
ax_left.set_ylabel("Acceleration [m/s²]")
ax_left.set_title("LEFT Sensor - Acceleration with Detected Peaks")
ax_left.legend()
ax_left.grid(True)

plt.tight_layout()
plt.show()
#"""
