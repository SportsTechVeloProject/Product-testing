# Shows the video and IMU together, can adjust so that they are played together
# Assumes IMU recoording begins before video
# Just looks at the y-komponent
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Parameters: -----------
imudata_raw = pd.read_csv('Python/time_adjusted_imu_100926_1.csv')
magdwick = pd.read_csv('Python/madgwick_right_data.csv')
video_path = 'DataSets/DataSet_100926/video1.mp4'  # Made for attempt 1

IMU_DATA_FREQ = 100
FIRST_SQUAT_IMU_TIME = 5.9
FIRST_SQUAT_VIDEO_TIME = 5.4
VIDEO_IMU_DIFF = FIRST_SQUAT_IMU_TIME - FIRST_SQUAT_VIDEO_TIME
# -----------


# Load the IMU data
imu_times = imudata_raw['t'].values
imu_y = imudata_raw['acc'].values

magdwick_times = magdwick['t'].values
magdwick_acc = magdwick['acc'].values

# Verify the SQUAT IMU TIME
for i in range(len(magdwick_acc)):
    if(magdwick_acc[i] >= 12): # arbitrary threshold
        print("Squat at: ", magdwick_times[i], " seconds")
        break

# Video setup
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Fps: ", fps)

# Initialize the plot could be extended to more axis?
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
ax1.set_title('Video Frame')
ax1.axis('off')

ax2.set_title('IMU Data Raw (y-axis)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Acceleration (m/s²)')
line1, = ax2.plot([], [], 'b-')
ax2.set_xlim(imu_times.min(), imu_times.max())
ax2.set_ylim(imu_y.min(), imu_y.max())

ax3.set_title('IMU Data Magdwick')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (m/s²)')
line2, = ax3.plot([], [], 'b-')
ax3.set_xlim(magdwick_times.min(), magdwick_times.max())
ax3.set_ylim(magdwick_acc.min(), magdwick_acc.max())

# Initialize the image for the video frame
img = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))


video_started = False
current_frame = 0
imudata_index = 0
# Function to update the plot and video frame
def update(frame_idx):
    global video_started, current_frame, imudata_index
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
    while (imu_times[imudata_index] <= current_time):
        line1.set_data(imu_times[:imudata_index+1], imu_y[:imudata_index+1])
        line2.set_data(magdwick_times[:imudata_index+1], magdwick_acc[:imudata_index+1])
        imudata_index += 1
    return img, line1, line2

# Create the animation
ani = FuncAnimation(fig, update, frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    interval=1000/fps, blit=False)

is_paused = False

def toggle_pause(event):
    global is_paused
    if event.key == ' ':
        if is_paused:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        is_paused = not is_paused

# Connect the key press event
fig.canvas.mpl_connect('key_press_event', toggle_pause)

plt.tight_layout()
plt.show()

# Release the video capture
cap.release()