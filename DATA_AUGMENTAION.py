import pandas as pd
import os
import numpy as np


def add_noise(data, noise_level=0.01):
    numeric_data = data.select_dtypes(include=[np.number])
    noise = np.random.normal(0, noise_level, numeric_data.shape)
    data.loc[:, numeric_data.columns] = numeric_data + noise
    return data
def create_new_participant_data(file_path, new_participant_id):
    try:
        df = pd.read_csv(file_path)
        df_noisy = add_noise(df)
        base_name = os.path.basename(file_path)
        new_file_name = base_name.replace('A', new_participant_id, 1)
        new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
        df_noisy.to_csv(new_file_path, index=False)
        print(f"New file created: {new_file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# List of new participants
new_participants = ['B', 'C', 'D', 'E']


bicep_directory = 'C:/Users/Dell/Downloads/BICEP'
for file_name in os.listdir(bicep_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(bicep_directory, file_name)
        for participant in new_participants:
            create_new_participant_data(file_path, participant)


tricep_directory = 'C:/Users/Dell/Downloads/tricep'
for file_name in os.listdir(tricep_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(tricep_directory, file_name)
        for participant in new_participants:
            create_new_participant_data(file_path, participant)
