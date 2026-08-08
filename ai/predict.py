import pandas as pd
import joblib


# ==================================================
# 1. Load Trained Model
# ==================================================

model_data = joblib.load(
    "ai/algorithm_recommender.pkl"
)

model = model_data["model"]
encoder = model_data["encoder"]
feature_columns = model_data["feature_columns"]


# ==================================================
# 2. Algorithm Information
# ==================================================

algorithm_info = {

    "Random Forest": {
        "description":
            "An ensemble learning algorithm that combines "
            "multiple decision trees. It is useful for "
            "classification problems with complex datasets."
    },

    "Decision Tree": {
        "description":
            "A tree-based algorithm that makes decisions "
            "using a sequence of simple rules. It is easy "
            "to understand and interpret."
    },

    "Logistic Regression": {
        "description":
            "A simple and efficient classification algorithm "
            "that works well for many binary and multiclass "
            "classification problems."
    },

    "SVM": {
        "description":
            "Support Vector Machine finds an effective "
            "decision boundary between classes and can "
            "work well with high-dimensional data."
    },

    "Naive Bayes": {
        "description":
            "A probabilistic algorithm based on Bayes' theorem. "
            "It is fast and can work well with smaller datasets."
    },

    "KNN": {
        "description":
            "K-Nearest Neighbors predicts a result based on "
            "the most similar nearby data points."
    },

    "Linear Regression": {
        "description":
            "A regression algorithm used to predict a numerical "
            "value based on relationships between variables."
    },

    "K-Means": {
        "description":
            "An unsupervised clustering algorithm that groups "
            "similar data points into clusters."
    }
}


# ==================================================
# 3. Display Header
# ==================================================

print("=" * 50)
print("       AI ALGORITHM RECOMMENDER")
print("=" * 50)

print("\nEnter your project details.\n")


# ==================================================
# 4. Input Validation Functions
# ==================================================

def get_choice(prompt, choices):

    while True:

        value = input(prompt).strip().title()

        if value in choices:
            return value

        print("\nInvalid input.")

        print("Please choose one of:")

        for choice in choices:
            print(f"  - {choice}")

        print()


def get_positive_integer(prompt):

    while True:

        value = input(prompt).strip()

        try:

            number = int(value)

            if number > 0:
                return number

            print(
                "Please enter a number greater than 0."
            )

        except ValueError:

            print(
                "Invalid number. "
                "Please enter a whole number."
            )


# ==================================================
# 5. Get User Input
# ==================================================

problem_type = get_choice(
    "Problem Type (Classification/Regression/Clustering): ",
    [
        "Classification",
        "Regression",
        "Clustering"
    ]
)


dataset_size = get_choice(
    "Dataset Size (Small/Medium/Large): ",
    [
        "Small",
        "Medium",
        "Large"
    ]
)


num_features = get_positive_integer(
    "Number of Features: "
)


num_records = get_positive_integer(
    "Number of Records: "
)


accuracy_priority = get_choice(
    "Accuracy Priority (Low/Medium/High): ",
    [
        "Low",
        "Medium",
        "High"
    ]
)


speed_priority = get_choice(
    "Speed Priority (Low/Medium/High): ",
    [
        "Low",
        "Medium",
        "High"
    ]
)


interpretability = get_choice(
    "Interpretability (Low/Medium/High): ",
    [
        "Low",
        "Medium",
        "High"
    ]
)


# ==================================================
# 6. Create User DataFrame
# ==================================================

user_data = pd.DataFrame([
    {
        "problem_type": problem_type,
        "dataset_size": dataset_size,
        "num_features": num_features,
        "num_records": num_records,
        "accuracy_priority": accuracy_priority,
        "speed_priority": speed_priority,
        "interpretability": interpretability
    }
])


# ==================================================
# 7. Define Feature Columns
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
# 8. Encode Categorical Features
# ==================================================

encoded_data = encoder.transform(
    user_data[categorical_columns]
)


encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(
        categorical_columns
    )
)


# ==================================================
# 9. Combine Features
# ==================================================

processed_input = pd.concat(
    [
        user_data[numerical_columns].reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ],
    axis=1
)


# Make sure the feature order is identical
# to the order used during model training.

processed_input = processed_input.reindex(
    columns=feature_columns,
    fill_value=0
)


# ==================================================
# 10. Predict Algorithm
# ==================================================

prediction = model.predict(
    processed_input
)


recommended_algorithm = prediction[0]


# ==================================================
# 11. Calculate Model Confidence
# ==================================================

probabilities = model.predict_proba(
    processed_input
)[0]


confidence = max(probabilities) * 100


# ==================================================
# 12. Get Algorithm Description
# ==================================================

description = algorithm_info[
    recommended_algorithm
]["description"]


# ==================================================
# 13. Display Recommendation
# ==================================================

print("\n" + "=" * 50)
print("           AI RECOMMENDATION")
print("=" * 50)


print(
    f"\nRecommended Algorithm:"
    f"\n{recommended_algorithm}"
)


print(
    f"\nModel Confidence:"
    f"\n{confidence:.2f}%"
)


print("\nWhy this recommendation?")


print(
    f"✓ Problem Type: {problem_type}"
)


print(
    f"✓ Dataset Size: {dataset_size}"
)


print(
    f"✓ Number of Features: {num_features}"
)


print(
    f"✓ Number of Records: {num_records}"
)


print(
    f"✓ Accuracy Priority: {accuracy_priority}"
)


print(
    f"✓ Speed Priority: {speed_priority}"
)


print(
    f"✓ Interpretability: {interpretability}"
)


print("\nAlgorithm Description:")

print(description)


print("\n" + "=" * 50)