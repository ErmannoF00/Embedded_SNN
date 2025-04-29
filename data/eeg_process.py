import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from pathlib import Path


# Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')


def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=250.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)


# Load and preprocess OpenBCI EEG data
def load_openbci_data(filepath, fs=250, segment_duration=1.0):

    df = pd.read_csv(filepath, skiprows=4)  # Header starts at line 5
    df.columns = df.columns.str.strip()     # Strip spaces
    print(df.columns)
    print(df.shape)

    # Correct filtering: we use 'EXG Channel' not 'EEG'
    eeg_channels = [col for col in df.columns if 'EXG Channel' in col]

    segment_len = int(fs * segment_duration)
    num_segments = len(df) // segment_len
    segments = []

    for i in range(num_segments):
        segment = []
        for ch in eeg_channels:
            raw = df[ch].values[i * segment_len:(i + 1) * segment_len]

            if len(raw) != segment_len:
                continue

            filtered = bandpass_filter(raw)
            if np.std(filtered) == 0:  # Avoid divide-by-zero
                continue

            normalized = (filtered - np.mean(filtered)) / np.std(filtered)
            segment.append(normalized)

        # Stack only if all channels are valid
        if len(segment) == len(eeg_channels):
            stacked = np.stack(segment, axis=1).flatten()
            segments.append(stacked)

    return segments



def save_segments(segments, output_folder="data", filename="simulate_input.txt"):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    file_path = Path(output_folder) / filename
    
    with open(file_path, 'w') as f:
        for idx, seg in enumerate(segments):
            label = 0 if idx % 2 == 0 else 1  # Simulated label
            line = ','.join([str(label)] + [str(x) for x in seg])
            f.write(line + '\n')
    print(f"Saved {len(segments)} segments to {file_path}")



# Example usage
if __name__ == "__main__":
    eeg_path = Path("data/OpenBCI_EEG/OpenBCI_GUI/data/EEG_Sample_Data/OpenBCI_GUI-v6-meditation.txt")
    segments = load_openbci_data(eeg_path)
    save_segments(segments)