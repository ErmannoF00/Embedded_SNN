# File: data/generate_dvs_gestures.py

import numpy as np

def generate_dvs_data(filename, num_gestures=100, num_events_per_gesture=5000, width=128, height=128):
    """
    Generate synthetic DVS data for gesture recognition.

    Args:
    - filename (str): Output file path to save the data
    - num_gestures (int): Number of synthetic gestures (gesture types)
    - num_events_per_gesture (int): Number of events per gesture
    - width (int): Width of the simulated DVS camera
    - height (int): Height of the simulated DVS camera
    """
    # Create a container for all gestures (num_gestures, num_events_per_gesture, 4)
    # 4 fields per event: timestamp (float), x-coordinate (int), y-coordinate (int), polarity (0 or 1)
    all_gestures = []
    
    for _ in range(num_gestures):
        gesture = []
        for _ in range(num_events_per_gesture):
            timestamp = np.random.rand() * 1000  # Random timestamp in milliseconds
            x = np.random.randint(0, width)  # Random x-coordinate
            y = np.random.randint(0, height)  # Random y-coordinate
            polarity = np.random.randint(0, 2)  # Random polarity (0 = off, 1 = on)
            gesture.append([timestamp, x, y, polarity])
        all_gestures.append(np.array(gesture))

    # Convert to a numpy array
    all_gestures = np.array(all_gestures)
    
    # Save data as a binary file
    all_gestures.tofile(filename)
    print(f"Simulated DVS gestures saved to {filename}")

# Example usage:
generate_dvs_data('data/dvs_gestures.bin')


# ✅ data/data.py (CREATE FILE IF MISSING)
# -------------------------
import numpy as np

def load_binary_dvs_data(path, shape=(100, 5000, 4)):
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)

# Usage
# data = load_binary_dvs_data("data/dvs_gestures.bin")