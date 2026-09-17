import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pykalman import KalmanFilter

file_path = 'DataSets/DataSet_150926/test1_gymaware_150926.csv'
data = pd.read_csv(file_path)

# Convert the 'time' column to datetime
data['time'] = pd.to_datetime(data['time'])

# Calculate the time difference in seconds from the first timestamp
data['time_seconds'] = (data['time'] - data['time'].iloc[0]).dt.total_seconds()
data['distanceM'] = data['distanceCm'] / 100.0

t = data['time_seconds'].to_numpy()
distance = data['distanceM'].to_numpy()
velocity = np.gradient(distance, t)
data['velocity'] = velocity
print(max(velocity), " ", min(velocity))

#acc = np.gradient(velocity, t)

data.to_csv('Python/Gymaware150926vel.csv')

plt.plot(t, distance, label='distance')
plt.plot(t, velocity, label='velocity')
plt.legend()
#plt.plot(t, acc)
plt.show()

velocity_spike_remove = velocity.copy()
#remove spikes
for i in range(1, len(velocity_spike_remove)):
    if abs(velocity_spike_remove[i]) >= 4:
        velocity_spike_remove[i] = velocity_spike_remove[i-1]
plt.title("Removed spikes >= 4")
plt.plot(t, velocity_spike_remove)
plt.show()

window_size = 25
smoothed_data_spike_removed = np.convolve(velocity_spike_remove, np.ones(window_size)/window_size, mode='same')
smoothed_data_original = np.convolve(velocity, np.ones(window_size)/window_size, mode='same')
plt.plot(t, smoothed_data_original, label='original data')
plt.plot(t, smoothed_data_spike_removed, label='spikes >= 4 removed first')
plt.legend()
plt.show()



# Unsucessfull attemps...
'''
dt = np.diff(t)
position = velocity.copy()
# Kalman Filter parameters
initial_state = np.array([position[0], 0])  # [position, velocity]
initial_covariance = np.eye(2)  # Initial covariance matrix
process_noise_cov = np.eye(2) * 0.1  # Process noise covariance
observation_noise_cov = np.array([[0.5]])  # Observation noise covariance

# Initialize lists to store smoothed states
smoothed_positions = []
smoothed_velocities = []

# Run the Kalman Filter for each time step
current_state = initial_state
current_covariance = initial_covariance

for i in range(len(position)):
    # Update transition matrix with current dt
    if i == 0:
        transition_matrix = np.eye(2)
    else:
        transition_matrix = np.array([[1, dt[i-1]], [0, 1]])

    # Predict step
    predicted_state = transition_matrix @ current_state
    predicted_covariance = transition_matrix @ current_covariance @ transition_matrix.T + process_noise_cov

    # Update step
    observation_matrix = np.array([[1, 0]])
    kalman_gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(
        observation_matrix @ predicted_covariance @ observation_matrix.T + observation_noise_cov
    )
    current_state = predicted_state + kalman_gain @ (position[i] - observation_matrix @ predicted_state)
    current_covariance = (np.eye(2) - kalman_gain @ observation_matrix) @ predicted_covariance

    # Store smoothed state
    smoothed_positions.append(current_state[0])
    smoothed_velocities.append(current_state[1])

# Convert to numpy arrays
smoothed_positions = np.array(smoothed_positions)
smoothed_velocities = np.array(smoothed_velocities)

plt.plot(t, smoothed_velocities)
plt.show()
'''
'''
def gaussian_smooth(t, y, sigma):
    smoothed = np.zeros_like(y)
    for i, ti in enumerate(t):
        weights = np.exp(-0.5 * ((t - ti) / sigma) ** 2)
        smoothed[i] = np.sum(weights * y) / np.sum(weights)
    return smoothed

sigma = 0.1  # smoothing strength
y_smoothed = gaussian_smooth(t, velocity, sigma)
plt.plot(t, y_smoothed)
plt.show()

'''
'''
from scipy.interpolate import interp1d
import numpy as np

# Interpolate :)
fs = 100
t_regular = np.linspace(t.min(), t.max(), num=int((t.max()-t.min())*fs))  # to 100Hz
f = interp1d(t, velocity, kind='cubic')  # Interpolate
v_regular = f(t_regular)

plt.plot(t, velocity, color='red')
plt.plot(t_regular, v_regular, color='blue')
plt.show()

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

cutoff = 5  # Hz
b, a = butter_lowpass(cutoff, fs)
filtered_signal = filtfilt(b, a, velocity)

plt.plot(t, filtered_signal)
plt.show()


'''
