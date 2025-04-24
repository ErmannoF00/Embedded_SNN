import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

# ----------------------------------------------
# 1. FILTER + LOADER
# ----------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=250.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def load_openbci_sample_data(filepath, channels=['EXG Channel 0', 'EXG Channel 1'], fs=250.0, segment_duration=1.0):
    df = pd.read_csv(filepath, skiprows=5)
    segment_len = int(fs * segment_duration)
    num_segments = len(df) // segment_len
    eeg_segments = []

    for i in range(num_segments):
        segment = []
        for ch in channels:
            raw_signal = df[ch].values[i * segment_len:(i + 1) * segment_len]
            filtered = bandpass_filter(raw_signal, fs=fs)
            normed = (filtered - np.mean(filtered)) / np.std(filtered)
            segment.append(normed)
        eeg_segments.append(np.stack(segment, axis=1))

    return eeg_segments  # List of np.array, shape (segment_len, n_channels)
