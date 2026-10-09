import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import MagdwickFilter
import VelocityCalculations

# ============================================================
# Read CSV
# ============================================================

file_path = os.path.join(
    os.path.dirname(__file__),
    "11squats011026.csv"
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

sample_rate = 104.0  # Hz

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

# ============================================================
# Calculate velocity using the filtered acceleration
# ============================================================

WINDOW_SIZE = 15

results_right = VelocityCalculations.calculate_velocity3(
    magnitude_right_filtered,
    t_right_sec,
    reps=11,
    window_size=WINDOW_SIZE
)

results_left = VelocityCalculations.calculate_velocity3(
    magnitude_left_filtered, 
    t_left_sec, 
    reps=11,
    window_size=WINDOW_SIZE
)



average_right = results_right['average_velocities']
peak_right = results_right['peak_velocities']
average_left = results_left['average_velocities']
peak_left = results_left['peak_velocities']



#Calculate average and peak velocities for each repetition
average = (np.array(average_left) + np.array(average_right)) / 2
peak = (np.array(peak_left) + np.array(peak_right)) / 2


# Print results:
for i in range(len(average_right)):
    print("Right Rep: ", i+1, " average: ", float(average_right[i]), " peak: ", float(peak_right[i]))
    print("Left Rep: ", i+1, " average: ", float(average_left[i]), " peak: ", float(peak_left[i]))
    print("Average Rep: ", i+1, " average: ", float(average[i]), " peak: ", float(peak[i]))
    print("----")

# ============================================================
# Save results to Excel
# ============================================================

results_df = pd.DataFrame({
    "Rep": np.arange(1, len(average) + 1),
    "Right average": average_right,
    "Right peak": peak_right,
    "Left average": average_left,
    "Left peak": peak_left,
    "Mean average (L+R)/2": average,
    "Mean peak (L+R)/2": peak,
}).round(4)

output_path = os.path.join(os.path.dirname(__file__), "velocity_results.xlsx")
results_df.to_excel(output_path, index=False)

print("Results saved to", output_path)
