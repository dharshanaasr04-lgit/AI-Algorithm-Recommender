import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the training dataset
data = pd.read_csv("ai/training_data.csv")

print("=== Original Dataset ===")
print(data.head())

# Separate input features and target
X = data.drop("recommended_algorithm", axis=1)
y = data["recommended_algorithm"]

# Categorical columns
categorical_columns = [
    "problem_type",
    "dataset_size",
    "accuracy_priority",
    "speed_priority",
    "interpretability"
]

# Numerical columns
numerical_columns = [
    "num_features",
    "num_records"
]

# One-hot encode categorical features
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(X[categorical_columns])

# Convert encoded data into a DataFrame
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Combine numerical and encoded features
processed_data = pd.concat(
    [
        X[numerical_columns].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ],
    axis=1
)

print("\n=== Processed Features ===")
print(processed_data.head())

print("\n=== Feature Names ===")
print(processed_data.columns.tolist())

print("\n=== Target Values ===")
print(y.value_counts())
