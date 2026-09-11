# Shows the video and IMU together, can adjust so that they are played together
# Assumes IMU recoording begins before video
# Just looks at the y-komponent
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Load the adjusteds IMU data
resampled_df = pd.read_csv('adjusted_resampled_imu_data_30Hz.csv')
imu_times = resampled_df['t'].values
imu_y = resampled_df['y'].values

video_path = 'video1.mp4'  # Made for attempt 1
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

FIRST_SQUAT_IMU_TIME = 5.8
FIRST_SQUAT_VIDEO_TIME = 5.2
VIDEO_IMU_DIFF = FIRST_SQUAT_IMU_TIME - FIRST_SQUAT_VIDEO_TIME

# Initialize the plot could be extended to more axis?
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_title('Video Frame')
ax1.axis('off')
ax2.set_title('IMU Data (y-axis)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Acceleration')
line, = ax2.plot([], [], 'b-')
ax2.set_xlim(imu_times.min(), imu_times.max())
ax2.set_ylim(imu_y.min(), imu_y.max())

# Initialize the image for the video frame
img = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))


video_started = False
current_frame = 0
# Function to update the plot and video frame
def update(frame_idx):
    global video_started, current_frame
    current_time = frame_idx / fps

    #Pause the video until syncronized.
    if(current_time >= VIDEO_IMU_DIFF and not video_started):
        video_started = True
        current_frame = 0
    
    if(video_started):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.set_array(frame)
            ax1.set_title(f'Video Frame: {current_frame}')
            current_frame +=1

    # Update the IMU plot
    line.set_data(imu_times[:frame_idx+1], imu_y[:frame_idx+1])
    return img, line

# Create the animation
ani = FuncAnimation(fig, update, frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    interval=1000/fps, blit=False)

plt.tight_layout()
plt.show()

# Release the video capture
cap.release()