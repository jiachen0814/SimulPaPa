""" 
Description: 
Author: Jiachen Li
Date: 2024-06-19
""" 
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
diseases = ['Use assisted device', 'uncontrolled heart', 'stroke', 'high blood pressure', 'diabetes', 'osteoporosis', 'take more than four medications']
# Define the base directory containing the folders
base_directory = os.getcwd()+'/.Users/cafe932a-4b40-316b-b15e-5d8a49c54bb2/agents'
print(os.getcwd())
# Initialize an empty list to store the data
data_list = []

def select_rows_with_distribution(df, target_distribution, num_rows=10, max_iterations=10000):
    best_selection = None
    best_fitness = np.inf
    fitness = 0
    # Iterate to find the best selection of rows
    for _ in range(max_iterations):
        # Randomly select num_rows from df
        selected_rows = df.sample(n=num_rows, replace=True)
        
        # Calculate mean and std for each column
        means = selected_rows.mean()
        stds = selected_rows.std()
        fitness = 0
        # Calculate fitness (distance) from target_distribution
        for col in df.columns:
            fitness = fitness + (abs(means[col] - target_distribution[col]['mean']) +(abs(stds[col] - target_distribution[col]['std']) if 'std' in target_distribution[col] else 0)/3)*target_distribution[col]['weight']
        
        # for col in df.columns:
        #     if target_distribution[col]['type'] == 'category':
        #         if means[col]!=target_distribution[col]['mean']:
        #             fitness = np.inf
        #             break
        #     fitness = fitness + (abs(means[col] - target_distribution[col]['mean']) +(abs(stds[col] - target_distribution[col]['std']) if 'std' in target_distribution[col] else 0))/target_distribution[col]['mean']
        
        # Update best selection if current selection is better
        if fitness < best_fitness:
            best_fitness = fitness
            best_selection = selected_rows.copy()
    # Print mean and standard deviation in a readable format
    for column in best_selection.columns:
        print(f"Column: {column}")
        print(f"Mean: {best_selection.mean()[column]:.2f}")
        print(f"Standard Deviation: {best_selection.std()[column]:.2f}")
        print("-" + "-" * len(column))
    return best_selection

def save_dataframe(df, base_filename,directory):
    # Initialize the counter
    counter = 0
    filename = f"{base_filename}.csv"
    
    # Check if the file already exists
    while os.path.isfile(os.path.join(directory,filename)):
        counter += 1
        filename = f"{base_filename}_{counter}.csv"
    
    # Save the DataFrame to the file
    df.to_csv(os.path.join(directory,filename), index=False)
    print(f"File saved as {filename}")



# Iterate through each folder in the base directory
for folder_name in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder_name)
    if os.path.isdir(folder_path):
        json_file_path = os.path.join(folder_path, 'info.json')
        if os.path.isfile(json_file_path):
            with open(json_file_path, 'r') as json_file:
                data_dict = json.load(json_file)
                data_dict['agent_num'] = int(folder_name)  # Add folder name to the dictionary
                data_list.append(data_dict)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)

# Set the folder name as the index
df.set_index('agent_num', inplace=True)
df = df.sort_values(by='agent_num')
# Change some data format
for disease in diseases:
    df[disease.replace(' ', '_').lower()] = df['disease'].apply(lambda x: 1 if disease.lower() in x.lower() else 0)
df['female'] = df['gender'].apply(lambda x: 1 if "female" in x.lower() else 0)
df['weight'] = df['weight'].apply(lambda x: float(x.split()[0]))
df['BMI'] = df['BMI'].apply(lambda x: float(x))
# # Save the DataFrame to a CSV file
# csv_file_path = 'all_info.csv'
# df.to_csv(csv_file_path)
# print(f"DataFrame saved to {csv_file_path}")


# Target mean and standard deviation for each column
target_distribution = {
    'female': {'mean': 8, 'std': 80, 'weight':1/8},
    'age': {'mean': 70.55, 'std': 7.5, 'weight':1/80},
    'BMI': {'mean': 30.15, 'std': 7.0, 'weight':1/30},
    'weight': {'mean': 179.95, 'std': 42.2, 'weight':1/180},
    'use_assisted_device': {'mean': 0.6, 'weight':1},
    'uncontrolled_heart': {'mean': 0.4, 'weight':1},
    'stroke': {'mean': 0.1, 'weight':1},
    'high_blood_pressure': {'mean': 0.6, 'weight':1},
    'diabetes': {'mean': 0.3, 'weight':1},
    'osteoporosis': {'mean': 0.6, 'weight':1},
    'take_more_than_four_medications': {'mean': 0.3, 'weight':1}
}

columns_to_consider = list(target_distribution.keys())
df_subset = df[columns_to_consider]
# Call the function to get selected rows
selected_rows_num = select_rows_with_distribution(df_subset, target_distribution, num_rows=10).index.values.tolist()
control_10 = df.iloc[selected_rows_num]

# # Save the DataFrame to a CSV file
# csv_file_path = os.path.join(base_directory, 'control_10.csv')
# control_10.to_csv(csv_file_path)
# print(f"DataFrame saved to {csv_file_path}")

save_dataframe(control_10, "control_10",base_directory)