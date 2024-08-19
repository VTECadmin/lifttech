import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd


# This class removes the high frequency data (that might be considered noise) from the data.
class LowPassFilter:
    def low_pass_filter(
            self,
            data_table,
            col,
            sampling_frequency,
            cutoff_frequency,
            order=5,
            phase_shift=True,
    ):
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq
        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
        return data_table


# This class contains methods for applying Principal Component Analysis (PCA) to the data.
class PrincipalComponentAnalysis:
    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                    data_table[col].max() - data_table[col].min()
            )
        return dt_norm

    def determine_pc_explained_variance(self, data_table, cols):
        dt_norm = self.normalize_dataset(data_table, cols)
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        return self.pca.explained_variance_ratio_

    def apply_pca(self, data_table, cols, number_comp):
        dt_norm = self.normalize_dataset(data_table, cols)
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])
        new_values = self.pca.transform(dt_norm[cols])
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]
        return data_table


# Example 1: Applying Low-Pass Filter

# Generate sample data
np.random.seed(0)
time = np.linspace(0, 1, 500)
data = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 50 * time)
data = data + np.random.normal(0, 0.5, len(time))  # Adding noise

# Create DataFrame
df = pd.DataFrame({"time": time, "signal": data})

# Apply Low-Pass Filter
lpf = LowPassFilter()
df_filtered = lpf.low_pass_filter(df, col="signal", sampling_frequency=500, cutoff_frequency=10, order=5)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['signal'], label='Original Signal')
plt.plot(df_filtered['time'], df_filtered['signal_lowpass'], label='Low-Pass Filtered Signal', linewidth=2)
plt.legend()
plt.title('Effect of Low-Pass Filter on Noisy Signal')
plt.show()

# Example 2: Applying Principal Component Analysis (PCA)

# Generate sample data
np.random.seed(0)
data = np.random.rand(100, 3)  # 100 samples, 3 features

# Create DataFrame
df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])

# Apply PCA
pca = PrincipalComponentAnalysis()
explained_variance = pca.determine_pc_explained_variance(df, ["feature1", "feature2", "feature3"])

# Print explained variance ratios
print("Explained Variance Ratios:", explained_variance)

# Apply PCA and transform data
df_pca = pca.apply_pca(df, ["feature1", "feature2", "feature3"], number_comp=2)

# Display the DataFrame with PCA components
print(df_pca.head())
