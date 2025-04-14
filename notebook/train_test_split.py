import pandas as pd
import numpy as np
# Example dataset
# Set random state manually
RANDOM_STATE = 42
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [6, 7, 8, 9, 10],
    'Label': [0, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())

# Shuffle the dataset
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Define train-test split ratio
train_ratio = 0.85
test_ratio = 0.15
train_size = int(len(df) * train_ratio)

# Split the dataset
train_data = df[:train_size]
test_data = df[train_size:]

# Display the splits
print("Train Data:")
print(train_data)
print("\nTest Data:")
print(test_data)
