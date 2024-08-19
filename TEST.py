import pandas as pd
from glob import glob

# Load specific accelerometer and gyroscope CSV files
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# List all CSV files in the specified directory
files = glob("../../data/raw/MetaMotion/*.csv")

# Define data path for consistent access
data_path = "../../data/raw/MetaMotion/"

# Initialize empty dataframes for accelerometer and gyroscope data
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

# Initial set numbers for accelerometer and gyroscope data
acc_set = 1
gyr_set = 1

# Loop over all files to read and categorize data
for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    # Distinguish between accelerometer and gyroscope data
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])

# Convert timestamp columns to pandas datetime format
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# Remove redundant columns
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]
del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# Reset index to ensure uniqueness
acc_df.reset_index(drop=True, inplace=True)
gyr_df.reset_index(drop=True, inplace=True)

# Merge accelerometer and gyroscope datasets
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# Print all unique labels and their counts before resampling
print("Unique labels before resampling:", data_merged['label'].unique())
print("Label counts before resampling:\n", data_merged['label'].value_counts())

# Resample data based on frequency specifications, ensuring no labels are lost
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

resampled_data = []
for label, group in data_merged.groupby('label'):
    if not group.empty:
        print(f"Resampling data for label: {label} with {len(group)} entries")
        group.index = pd.to_datetime(group.index, unit='ms')  # Ensure the index is a DatetimeIndex
        resampled_group = group.resample(rule="200ms").apply(sampling).interpolate(method='linear').ffill().bfill()
        resampled_data.append(resampled_group)

data_resampled = pd.concat(resampled_data)
data_resampled["set"] = data_resampled["set"].astype("int")

# Print all unique labels and their counts after resampling
print("Unique labels after resampling:", data_resampled['label'].unique())
print("Label counts after resampling:\n", data_resampled['label'].value_counts())

# Save the processed data to a new file
data_resampled.to_pickle("../../data/interim/03_data_processed.pkl")

print("Unique labels:", data_resampled['label'].unique())
print("Label counts:\n", data_resampled['label'].value_counts())
