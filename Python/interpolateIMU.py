# Uses one of the right / left data and interpolates it to 30Hz, also makes it begin at 0
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Load the original IMU data
data = pd.read_csv('right_data.csv')

# Extract time and IMU data
t = data['t'].values
imu_data = data[['x', 'y', 'z', 'gx', 'gy', 'gz']].values

# Adjust the IMU timestamps
adjusted_t = t - t[0]  # Subtract the offset to align with the video

start_time = adjusted_t.min()
end_time = adjusted_t.max()
target_times = np.arange(start_time, end_time, 1/30)

# Interpolate each IMU column to 30 Hz
resampled_data = np.zeros((len(target_times), imu_data.shape[1]))
for i in range(imu_data.shape[1]):
    interp_func = interp1d(adjusted_t, imu_data[:, i], kind='linear', fill_value='extrapolate')
    resampled_data[:, i] = interp_func(target_times)

# Create a DataFrame for the adjusted and resampled data
resampled_df = pd.DataFrame(resampled_data, columns=['x', 'y', 'z', 'gx', 'gy', 'gz'])
resampled_df['t'] = target_times

# Save the adjusted and resampled data to a new CSV file
resampled_df.to_csv('adjusted_resampled_imu_data_30Hz.csv', index=False)