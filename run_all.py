import os

print("[1] Preprocessing EEG data...")
os.system("python data/eeg_process.py")

print("[2] Training AdEx SNN...")
os.system("python scripts/train.py")

print("[3] Exporting weights for C++...")
os.system("python scripts/export_weights.py")

print("✅ Full pipeline completed.")