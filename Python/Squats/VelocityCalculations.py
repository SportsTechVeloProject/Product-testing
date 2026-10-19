import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import ValidatePeaks as ValidatePeaks

# Different zone thresholds
threshold_dict = {
    'slow' : [3, 312],
    'medium': [2, 208],
    'fast': [3, 208]
}

def peak_detect(velocity, start, stop, reps, type):
    speed = 3.0
    peaks, peak_info = find_peaks(velocity[start:stop], height=speed, distance=threshold_dict[type][1])
    i = 0
    while(len(peaks) < reps):
        speed = speed - 0.01
        peaks, peak_info = find_peaks(velocity[start:stop], height=speed, distance=threshold_dict[type][1])
        i += 1
    print(i)
    if(len(peaks) > reps):
        peaks = peaks[:-1]
    return peaks + start, peak_info

def find_drift(velocity, t_sec, magnitude_raw, type):
    Treshold = threshold_dict[type][0] * np.mean(magnitude_raw)
    start_index = 520
    indexes, _ = find_peaks(magnitude_raw[start_index:], height=Treshold)
    print(indexes)

    stop_index = indexes[-1]

    stop_index =  start_index + stop_index - 208
    print("stop", stop_index)
    
    #k, m = np.polyfit(t_sec[start:stop], velocity[start:stop], 1)
    end = np.mean(velocity[stop_index-208:stop_index])
    start = np.mean(velocity[52:260])

    k = (end - start) / (t_sec[stop_index-104] - t_sec[156])

    return k, start, start_index, stop_index

   

def calculate_velocity3(magnitude_filtered, t_sec, type, reps, magnitude_raw, window_size=7):
    all_repetition_acceleration = []
    all_repetition_time = []
    all_repetition_velocity = []

    average_velocities = []

    # Filter acceleration with moving average:
    magnitude_filtered = np.convolve(magnitude_filtered, np.ones(window_size) / window_size, 'same') 

    #Average of the acceleration on 1-2s
    mean_acceleration = np.mean(magnitude_filtered[104:208])


    # Calculate the velocity by integration
    velocity = cumulative_trapezoid(magnitude_filtered - mean_acceleration, t_sec, initial=0)
    plt.plot(t_sec, velocity)
    plt.show()

    # Assume the drift is linear.
    k, m, start_index, stop_index = find_drift(velocity, t_sec, magnitude_raw, type)
    y = k*t_sec + m
    diff = velocity - y
    print("Stop index", stop_index)

    peaks, peak_info = peak_detect(diff, start_index, stop_index, reps, type)
    
    if (type == 'fast'):
        
        ind = peaks[0] - 416
        velocity = velocity[ind:]
        t_sec = t_sec[ind:]
        t_sec = t_sec - t_sec[0]
        k, m, start_index, stop_index = find_drift(velocity, t_sec, magnitude_raw[ind:], type)
        y = k*t_sec #+ m
        diff = velocity - y
        peaks, peak_info = peak_detect(diff, 0, stop_index, reps, type)

    crossings = []
    for peak in peaks:
        start = peak
        end = peak
        while(diff[start] > 0):
            start -= 1
        crossings.append(start)
        while(diff[end] > 0):
            end += 1
        crossings.append(end)
        mean_v = np.mean(diff[start:end])
        average_velocities.append(mean_v)
    
    plt.plot(t_sec, y)
    plt.plot(t_sec, velocity)
    plt.plot(t_sec[peaks], diff[peaks], 'x', markersize=10)
    plt.plot(t_sec, diff)
    plt.show()
       
    return {
        "all_acceleration": all_repetition_acceleration,
        "all_time": all_repetition_time,
        "all_velocity": all_repetition_velocity,
        "average_velocities": average_velocities,
        "peak_velocities": peak_info['peak_heights'],
        "crossing": crossings,
        "diff": diff,
    }

def find_drift2(velocity, t_sec):

    k, m = np.polyfit(t_sec, velocity, 1)

    return k, m

def peak_detect_external_start_stop(velocity, reps, type):
    speed = 3.0
    peaks, peak_info = find_peaks(velocity, height=speed, distance=threshold_dict[type][1])
    i = 0
    while(len(peaks) < reps):
        speed = speed - 0.01
        peaks, peak_info = find_peaks(velocity, height=speed, distance=threshold_dict[type][1])
        i += 1
    print(i)
    if(len(peaks) > reps):
        if peaks[0] > peaks[-1]:
            peaks = peaks[:-1]
        else:
            peaks = peaks[1:]
    return peaks, peak_info

def find_average(peaks, velocity):
    crossings = []
    average_vel = []
    print("pekas", peaks)

    for peak in peaks:
        start = peak
        end = peak
        while(velocity[start] > 0):
            start -= 1
        crossings.append(start)
        while(velocity[end] > 0):
            end += 1
            if(end > (len(velocity) -1)):
                break
        crossings.append(end)
        average_vel.append(np.mean(velocity[start:end]))
    return crossings, average_vel



def calculate_velocity_external_start_stop(magnitude_filtered_right, magnitude_filtered_left, t_sec_right, t_sec_left, type, reps, window_size=7):
    all_repetition_acceleration = []
    all_repetition_time = []
    all_repetition_velocity = []
    all_repetition_peak = []
    all_repetition_diff = []

    average_velocities = []

    # Filter acceleration with moving average:
    magnitude_filtered_right = np.convolve(magnitude_filtered_right, np.ones(window_size) / window_size, 'same') 
    magnitude_filtered_left = np.convolve(magnitude_filtered_left, np.ones(window_size) / window_size, 'same') 

    #Average of the acceleration on 1-2s
    mean_acceleration_right = np.mean(magnitude_filtered_right[0:52])
    mean_acceleration_left = np.mean(magnitude_filtered_left[0:52])


    # Calculate the velocity by integration
    velocity_right = cumulative_trapezoid(magnitude_filtered_right - mean_acceleration_right, t_sec_right, initial=0)
    velocity_left = cumulative_trapezoid(magnitude_filtered_left - mean_acceleration_left, t_sec_left, initial=0)

    # Assume the drift is linear.
    k_right, m_right = find_drift2(velocity_right, t_sec_right)
    k_left, m_left = find_drift2(velocity_left, t_sec_left)
    y_right = k_right*t_sec_right + m_right
    diff_right = velocity_right - y_right

    y_left = k_left*t_sec_left + m_left
    diff_left = velocity_left - y_left
    peaks_left, peak_info_left = peak_detect_external_start_stop(diff_left, reps, type)
    peaks_right, peak_info_right = peak_detect_external_start_stop(diff_right, reps, type)

    # Calculate the root-mean-square-error of the linear approximation.
    rmse_left = np.sqrt(np.mean(diff_left ** 2))
    rmse_right = np.sqrt(np.mean(diff_right ** 2))
    print("rmse_left ", rmse_left, " rmse_right ", rmse_right)

    # Pick the one with the smallest root mean square error
    if(rmse_left < rmse_right):
        crossings, average_velocities = find_average(peaks_left, diff_left)
        all_repetition_peak = peak_info_left['peak_heights']
        all_repetition_diff = diff_left
    else:
        crossings, average_velocities = find_average(peaks_right, diff_right)
        all_repetition_peak = peak_info_right['peak_heights']
        all_repetition_diff = diff_right

    plt.plot(t_sec_right, all_repetition_diff, label="corrected velocity")
    plt.plot(t_sec_right, velocity_right, label = "original velocity")
    plt.plot(t_sec_right, y_right, label = "drift", linestyle="dashed")
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.legend()
    print(crossings)
    #plt.plot(t_sec_left[crossings], all_repetition_diff[crossings], 'x', markersize=10)
    plt.show()
       
    return {
        "all_acceleration": all_repetition_acceleration,
        "all_time": all_repetition_time,
        "all_velocity": all_repetition_velocity,
        "average_velocities": average_velocities,
        "peak_velocities": all_repetition_peak,
        "crossing": crossings,
        "diff": all_repetition_diff,
    }

