import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. Load Dataset
# ==================================================

data = pd.read_csv("ai/training_data.csv")

print("=== Dataset Loaded ===")
print("Total records:", len(data))


# ==================================================
# 2. Separate Features and Target
# ==================================================

X = data.drop("recommended_algorithm", axis=1)

y = data["recommended_algorithm"]


# ==================================================
# 3. Define Columns
# ==================================================

categorical_columns = [
    "problem_type",
    "dataset_size",
    "accuracy_priority",
    "speed_priority",
    "interpretability"
]

numerical_columns = [
    "num_features",
    "num_records"
]


# ==================================================
# 4. Encode Categorical Features
# ==================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = encoder.fit_transform(
    X[categorical_columns]
)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(
        categorical_columns
    )
)


# ==================================================
# 5. Combine Features
# ==================================================

X_processed = pd.concat(
    [
        X[numerical_columns].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ],
    axis=1
)


print("\n=== Processed Dataset ===")
print("Features:", X_processed.shape[1])
print("Records:", X_processed.shape[0])


# ==================================================
# 6. Train/Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n=== Dataset Split ===")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==================================================
# 7. Create Random Forest Model
# ==================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==================================================
# 8. Train Model
# ==================================================

model.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully.")


# ==================================================
# 9. Make Predictions
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 10. Calculate Accuracy
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n=== Model Accuracy ===")
print(f"{accuracy * 100:.2f}%")


# ==================================================
# 11. Classification Report
# ==================================================

print("\n=== Classification Report ===")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==================================================
# 12. Confusion Matrix
# ==================================================

print("\n=== Confusion Matrix ===")

labels = sorted(
    y.unique()
)

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("Labels:")
print(labels)

print("\nMatrix:")
print(matrix)


# ==================================================
# 13. Save Model
# ==================================================

model_data = {
    "model": model,
    "encoder": encoder,
    "feature_columns": X_processed.columns.tolist()
}

joblib.dump(
    model_data,
    "ai/algorithm_recommender.pkl"
)

print("\nModel saved successfully.")

print(
    "File: ai/algorithm_recommender.pkl"
)