import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from pathlib import Path


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

def load_openbci_sample_data(filepath, fs=250.0, segment_duration=1.0):
    df = pd.read_csv(filepath, skiprows=5, delimiter=',', engine='python')  # Use ',' as delimiter
    print("Columns in file:", df.columns.tolist())

    # Manually select columns (assuming channels are in the 2nd and 3rd columns)
    channels = df.columns[1:3]  # Adjust according to your actual column layout
    
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

    return eeg_segments

def save_eeg_data_to_txt(segments, output_folder="data", filename="eeg_encoded_input.txt"):
    # Ensure the output folder exists
    output_folder_path = Path(__file__).resolve().parents[1] / output_folder
    output_folder_path.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

    # Create the full path for the output file
    file_path = output_folder_path / filename
    
    with open(file_path, 'w') as file:
        for segment in segments:
            for row in segment:
                file.write(','.join(map(str, row)) + '\n')  # Write each row in CSV format
    print(f"EEG data saved to {file_path}")

# Get the root directory of your project (assuming this file is in the root or a subfolder)
project_root = Path(__file__).resolve().parents[1] 

# Build the path to the EEG data file
eeg_data_path = project_root / "data" / "OpenBCI_EEG" / "OpenBCI_GUI" / "data" / "EEG_Sample_Data" / "OpenBCI_GUI-v6-meditation.txt"

# Load EEG data and process it
segments = load_openbci_sample_data(eeg_data_path)

# Save the EEG data to a .txt file in the 'data' folder
save_eeg_data_to_txt(segments)

# Print some details about the segments
print("Num segments:", len(segments))
print("Shape of first segment:", segments[0].shape)
