import pandas as pd
import numpy as np
import os
import imufusion
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# ============================================================
# FILTER FALSE PEAKS
# ============================================================

def validate_peaks(signal, peaks, sample_rate):
    """
    Removes peaks that look like short, unnatural spikes.

    signal       : acceleration signal
    peaks        : peaks detected by find_peaks()
    sample_rate  : sampling frequency in Hz
    """

    valid_peaks = []

    # Number of samples to inspect before/after peak
    BEFORE_TIME = 0.05      # seconds
    AFTER_TIME = 0.05        # seconds

    BEFORE_SAMPLES = int(BEFORE_TIME * sample_rate)
    AFTER_SAMPLES = int(AFTER_TIME * sample_rate)

    for peak in peaks:

        # Make sure we have enough data around the peak
        if peak < BEFORE_SAMPLES:
            continue

        if peak + AFTER_SAMPLES >= len(signal):
            continue

        # Values before and after peak
        before = signal[
            peak - BEFORE_SAMPLES:peak
        ]

        after = signal[
            peak + 1:peak + AFTER_SAMPLES
        ]

        peak_value = signal[peak]

        # ----------------------------------------------------
        # 1. Calculate how much the signal changes
        # ----------------------------------------------------

        before_range = np.max(before) - np.min(before)
        after_range = np.max(after) - np.min(after)

        # ----------------------------------------------------
        # 2. Check that peak is actually higher than
        #    the surrounding signal
        # ----------------------------------------------------

        baseline = np.mean(
            np.concatenate([
                before[-5:],
                after[:5]
            ])
        )

        peak_height = peak_value - baseline

        # ----------------------------------------------------
        # 3. Check that the signal is not changing
        #    extremely rapidly
        # ----------------------------------------------------

        before_diff = np.diff(before)
        after_diff = np.diff(after)

        max_change_before = np.max(np.abs(before_diff))
        max_change_after = np.max(np.abs(after_diff))

        # ----------------------------------------------------
        # Thresholds
        # ----------------------------------------------------

        MIN_PEAK_HEIGHT = -5.0

        MAX_CHANGE_BEFORE = 3.0
        MAX_CHANGE_AFTER = 3.0

        # ----------------------------------------------------
        # Validate peak
        # ----------------------------------------------------

        if (
            peak_height > MIN_PEAK_HEIGHT
            and max_change_before < MAX_CHANGE_BEFORE
            and max_change_after < MAX_CHANGE_AFTER
        ):

            valid_peaks.append(peak)

    return np.array(valid_peaks)
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


##Calcualtions with right data 
peaks_right, properties_right = find_peaks(
    magnitude_right_filtered,
    height=-5,
    distance=100
)

#calculations left data 
peaks_left, properties_left = find_peaks(
    magnitude_left_filtered,
    height=-5,
    distance=100
)

if(peaks_left.size != peaks_right.size):
    print("Warning: The number of peaks detected in left and right data is not equal.")
    print(f"Number of peaks in left data: {peaks_left.size}")
    print(f"Number of peaks in right data: {peaks_right.size}")
else:
    print(f"Number of peaks detected in both left and right data: {peaks_left.size}")

peak_times_left = t_left_sec[peaks_left]
peak_times_right = t_right_sec[peaks_right]

print("\nLEFT peak timestamps [s]:")
for idx, ts in zip(peaks_left, peak_times_left):
    print(f"  Sample {idx} -> {round(ts, 3)} s")

print("\nRIGHT peak timestamps [s]:")
for idx, ts in zip(peaks_right, peak_times_right):
    print(f"  Sample {idx} -> {round(ts, 3)} s")


peaks_right, properties_right = find_peaks(
    magnitude_right_filtered,
    height=-5,
    distance=100
)

# Filter false peaks
valid_peaks_right = validate_peaks(
    magnitude_right_filtered,
    peaks_right,
    sample_rate
)

print("Peaks before filtering:", len(peaks_right))
print("Peaks after filtering:", len(valid_peaks_right))


#============================================================
# Plot acceleration and detected peaks for RIGHT sensor
# ============================================================

plt.figure(figsize=(14, 7))

# Plot acceleration
plt.plot(
    t_right_sec,
    magnitude_right_filtered,
    label="Filtered acceleration"
)

# Plot detected peaks
plt.plot(
    t_right_sec[valid_peaks_right],
    magnitude_right_filtered[valid_peaks_right],
    "x",
    markersize=10,
    label="Detected peaks"
)

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.title("RIGHT Sensor - Acceleration with Detected Peaks")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


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


##Calcualtions with right data 
peaks_right, properties_right = find_peaks(
    magnitude_right_filtered,
    height=13,
    distance=50
)

#calculations left data 
peaks_left, properties_left = find_peaks(
    magnitude_left_filtered,
    height=13,
    distance=50
)

if(peaks_left.size != peaks_right.size):
    print("Warning: The number of peaks detected in left and right data is not equal.")
    print(f"Number of peaks in left data: {peaks_left.size}")
    print(f"Number of peaks in right data: {peaks_right.size}")
else:
    print(f"Number of peaks detected in both left and right data: {peaks_left.size}")

# ============================================================
# Movement detection
# ============================================================

WINDOW_SIZE = 10

START_THRESHOLD = 1.5

PAUSE_TIME = 0.3       # seconds
MIN_START_SAMPLES = 5

magnitude_left_windowed = pd.Series(
    magnitude_left_filtered
).rolling(
    window=WINDOW_SIZE,
    center=True
).mean()

magnitude_left_windowed = magnitude_left_windowed.fillna(0)


# ============================================================
# Convert pause time to number of samples
# ============================================================

PAUSE_SAMPLES = int(PAUSE_TIME * sample_rate)

print("Samples needed for pause:", PAUSE_SAMPLES)


# ============================================================
# Detect movement
# ============================================================

movement_starts = []
movement_pauses = []

state = "REST"

i = 0

while i < len(magnitude_left_windowed) - PAUSE_SAMPLES:

    # --------------------------------------------------------
    # REST -> MOVING
    # --------------------------------------------------------

    if state == "REST":

        if (
            magnitude_left_windowed.iloc[
                i:i + MIN_START_SAMPLES
            ] > START_THRESHOLD
        ).all():

            movement_starts.append(i)

            print(
                "Movement starts:",
                i,
                "Time:",
                round(t_left_sec[i], 2),
                "s"
            )

            state = "MOVING"

            i += MIN_START_SAMPLES
            continue


    # --------------------------------------------------------
    # MOVING -> REST
    # --------------------------------------------------------

    elif state == "MOVING":

        # Check if acceleration stays below the
        # movement threshold for 1 second

        if (
            magnitude_left_windowed.iloc[
                i:i + PAUSE_SAMPLES
            ] < START_THRESHOLD
        ).all():

            movement_pauses.append(i)

            print(
                "Movement pauses:",
                i,
                "Time:",
                round(t_left_sec[i], 2),
                "s"
            )

            state = "REST"

            i += PAUSE_SAMPLES
            continue

    i += 1
