import pandas as pd

# Load the training dataset
data = pd.read_csv("ai/training_data.csv")

# Display the complete dataset
print("=== AI Algorithm Recommender Dataset ===")
print(data)

# Display dataset information
print("\n=== Dataset Information ===")
print(data.info())

# Display number of rows and columns
print("\n=== Dataset Shape ===")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# Display column names
print("\n=== Columns ===")
print(data.columns.tolist())

# Check for missing values
print("\n=== Missing Values ===")
print(data.isnull().sum())

# Display the algorithms available in the dataset
print("\n=== Recommended Algorithms ===")
print(data["recommended_algorithm"].value_counts())