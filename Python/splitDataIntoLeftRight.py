# Splits file into left and right imu data, needs to be in same folder
# Results used in interpolateIMU.py
import pandas as pd
import os
import numpy as np

input_file_path = 'DataSets/DataSet_100926/Recording1_IMU.csv'

file_path = os.path.join(os.path.dirname(__file__), input_file_path)
data = pd.read_csv(input_file_path)

left_data = data[data['sensor'] == "left"]
right_data = data[data['sensor'] == "right"]

t_left = left_data["t"].values
imu_left = left_data[["x", "y", "z", "gx", "gy", "gz"]].values
start_left = t_left.min()
end_left = t_left.max()
left_times = np.arange(start_left, end_left, 1/30)


# Places the results in main folder?
left_data.to_csv('Python/left_data.csv', index=False)
right_data.to_csv('Python/right_data.csv', index=False)