import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# Load data
df = pd.read_pickle("../../data/interim/03_data_processed.pkl")
predictor_columns = list(df.columns[:6])

# Setting the plot style for better aesthetics
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Display basic information about the dataframe, including column types and non-null values
df.info()

# Print the number of unique exercises (labels) before any processing
unique_exercises = df["label"].nunique()
print("Number of unique exercises before processing:", unique_exercises)

# Dealing with missing values (imputation)
for col in predictor_columns:
    df[col] = df[col].interpolate()

# Ensure there are no NaN values in the DataFrame
df = df.dropna(subset=predictor_columns, how='any')

# Calculate the duration for each unique set and store it in a new column 'duration'
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    df.loc[df["set"] == s, "duration"] = duration.total_seconds()

# Calculate the mean duration for each 'category' and store it in 'duration_df'
duration_df = df.groupby(["category"])["duration"].mean()
print("Mean duration for first category divided by 5:", duration_df.iloc[0] / 5)
print("Mean duration for second category divided by 10:", duration_df.iloc[1] / 10)

# Print the number of unique exercises (labels) after handling missing values
unique_exercises = df["label"].nunique()
print("Number of unique exercises after handling missing values:", unique_exercises)

# Butterworth lowpass filter
df_lowpass = df.copy()
LowPass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.3

# Apply a low-pass filter on all predictor columns
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# Ensure there are no NaN values in df_lowpass before PCA
df_lowpass = df_lowpass.dropna()

# Print the number of unique exercises (labels) after low-pass filter
unique_exercises = df_lowpass["label"].nunique()
print("Number of unique exercises after low-pass filter:", unique_exercises)

# Principal component analysis PCA
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal component number")
plt.ylabel("explained Variance")
plt.show()

# Apply PCA to the data and keep only the first 3 principal components
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# Sum of squares attributes
df_squared = df_pca.copy()
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2
df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)
subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# Print the number of unique exercises (labels) after PCA and sum of squares
unique_exercises = df_squared["label"].nunique()
print("Number of unique exercises after PCA and sum of squares:", unique_exercises)

# Temporal abstraction
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = int(1000 / 200)

# Abstract the data using both mean and standard deviation over the defined window size
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

# Ensure there are no NaN values in df_temporal before frequency transformation
df_temporal = df_temporal.dropna()

# Print the number of unique exercises (labels) after temporal abstraction
unique_exercises = df_temporal["label"].nunique()
print("Number of unique exercises after temporal abstraction:", unique_exercises)

# Frequency features
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()
fs = int(1000 / 200)
ws = int(2800 / 200)

# Apply Fourier transformation to abstract the 'acc_y' frequency domain characteristics
df_freq["epoch (ms)"] = df_freq.index  # Retain the epoch (ms) before transformation

# Ensure no zero values in the data to prevent log errors
for col in predictor_columns:
    df_freq[col] = df_freq[col].replace(0, 1e-10)  # Replace zeros with a small value

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
print(df_freq.columns)

subset = df_freq[df_freq["set"] == 15]
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

# Frequency abstracted subsets
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset["epoch (ms)"] = df_freq[df_freq["set"] == s].index  # Retain the epoch (ms) before transformation
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# Print the number of unique exercises (labels) after frequency features and before dropping NaN values
unique_exercises = df_freq["label"].nunique()
print("Number of unique exercises after frequency features (before dropping NaN):", unique_exercises)

# Handle NaNs appropriately
df_freq = df_freq.fillna(method='ffill').fillna(method='bfill')

# Print label counts after handling NaN
print("Label counts after handling NaN:")
print(df_freq["label"].value_counts())

# Print the number of unique exercises (labels) after frequency features
unique_exercises = df_freq["label"].nunique()
print("Number of unique exercises after frequency features:", unique_exercises)

# Clustering
df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]

k_values = range(2, 10)
inertias = []
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    df_cluster[f'cluster_{k}'] = cluster_labels  # Add cluster labels to the DataFrame
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# Use the optimal number of clusters (assuming it's the one with the least inertia)
optimal_k = k_values[np.argmin(inertias)]
df_cluster['cluster'] = df_cluster[f'cluster_{optimal_k}']  # Use the optimal cluster labels

# 3D scatter plot visualization of the clusters based on accelerometer readings
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
colors = ["r", "g", "b", "y", "c"]

for idx, c in enumerate(df_cluster["cluster"].unique()):
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        color=colors[idx % len(colors)],
        label=f"Cluster {c}",
    )
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# 3D scatter plot visualization based on labels
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for idx, c in enumerate(df_cluster["label"].unique()):
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=f"label {c}",
    )
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Print the number of unique exercises (labels) after clustering
unique_exercises = df_cluster["label"].nunique()
print("Number of unique exercises after clustering:", unique_exercises)

# Export dataset
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
