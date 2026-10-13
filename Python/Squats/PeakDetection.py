##Detect Peaks Using moving average filter 
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def peak_detect(velocity, start, stop, reps):
    speed = 3.0
    peaks, _ = find_peaks(velocity[start:stop], height=speed, distance=52)
    while(len(peaks) < reps):
        speed -= 0.1
        peaks, _ = find_peaks(velocity[start:stop], height=speed, distance=52)
    return peaks