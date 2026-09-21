import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import ValidatePeaks as ValidatePeaks

def calculate_velocity(magnitude_filtered, t_sec, valid_peaks, window_size=7):
    

    all_repetition_acceleration = []
    all_repetition_time = []
    all_repetition_velocity = []

    average_velocities = []
    peak_velocities = []

    # Need at least two peaks to define one repetition

    if len(valid_peaks) < 2:

        print(
            "\nNot enough peaks to calculate "
            "repetitions."
        )

        return {
            "all_acceleration": all_repetition_acceleration,
            "all_time": all_repetition_time,
            "all_velocity": all_repetition_velocity,
            "average_velocities": average_velocities,
            "peak_velocities": peak_velocities,
        }

    for rep_number in range(len(valid_peaks)):

        # ----------------------------------------------------
        # Start and stop of this repetition
        # ----------------------------------------------------

        if rep_number == 0:
            # First repetition:
            # Start from the beginning of the measurement
            start = 0
            stop = valid_peaks[0]
        else:
            # Following repetitions:
            # Start at previous detected peak, stop at current peak
            start = valid_peaks[rep_number - 1]
            stop = valid_peaks[rep_number]

        # ----------------------------------------------------
        # Extract acceleration and time
        # ----------------------------------------------------

        repetition = magnitude_filtered[start:stop]
        repetition_time = t_sec[start:stop]

        # ----------------------------------------------------
        # Reverse the data
        # ----------------------------------------------------

        repetition = np.flip(repetition)
        repetition_time = np.flip(repetition_time)

        # ----------------------------------------------------
        # Find acceleration section
        # ----------------------------------------------------

        acceleration = []
        time_stamps = []

        for j in range(window_size, len(repetition) - window_size):

            previous_average = np.mean(repetition[j - window_size:j])
            next_average = np.mean(repetition[j:j + window_size])

            if next_average < previous_average:
                acceleration.append(repetition[j])
                time_stamps.append(repetition_time[j])
            else:
                break

        acceleration = np.array(acceleration)
        time_stamps = np.array(time_stamps)

        if len(acceleration) < 2:
            print(f"\nRepetition {rep_number}: Not enough acceleration data.")
            continue

        # ----------------------------------------------------
        # Calculate velocity
        # ----------------------------------------------------

        velocity = cumulative_trapezoid(acceleration, time_stamps, initial=0)
        average_velocity = np.mean(np.abs(velocity))
        peak_velocity = np.max(np.abs(velocity))

        all_repetition_acceleration.append(acceleration)
        all_repetition_time.append(time_stamps)
        all_repetition_velocity.append(velocity)
        average_velocities.append(average_velocity)
        peak_velocities.append(peak_velocity)

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(f"\nRepetition {rep_number}")
        print(f"Peak start: {t_sec[start]:.3f} s")
        print(f"Peak end: {t_sec[stop]:.3f} s")
        print(f"Acceleration samples: {len(acceleration)}")


    return {
        "all_acceleration": all_repetition_acceleration,
        "all_time": all_repetition_time,
        "all_velocity": all_repetition_velocity,
        "average_velocities": average_velocities,
        "peak_velocities": peak_velocities,
    }