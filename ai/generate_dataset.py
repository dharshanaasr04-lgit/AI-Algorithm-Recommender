import random
import pandas as pd

random.seed(42)

data = []

ALGORITHMS = [
    "Random Forest",
    "Decision Tree",
    "Logistic Regression",
    "SVM",
    "Naive Bayes",
    "KNN",
    "Linear Regression",
    "K-Means"
]


def create_row(
    problem_type,
    dataset_size,
    num_features,
    num_records,
    accuracy_priority,
    speed_priority,
    interpretability,
    algorithm
):
    return {
        "problem_type": problem_type,
        "dataset_size": dataset_size,
        "num_features": num_features,
        "num_records": num_records,
        "accuracy_priority": accuracy_priority,
        "speed_priority": speed_priority,
        "interpretability": interpretability,
        "recommended_algorithm": algorithm
    }


# --------------------------------------------------
# 1. Random Forest
# --------------------------------------------------

for _ in range(30):

    dataset_size = random.choice(["Medium", "Large"])
    num_features = random.randint(15, 50)

    num_records = (
        random.randint(5000, 15000)
        if dataset_size == "Medium"
        else random.randint(20000, 100000)
    )

    data.append(
        create_row(
            "Classification",
            dataset_size,
            num_features,
            num_records,
            "High",
            random.choice(["Medium", "Low"]),
            random.choice(["Medium", "Low"]),
            "Random Forest"
        )
    )


# --------------------------------------------------
# 2. Decision Tree
# --------------------------------------------------

for _ in range(30):

    dataset_size = random.choice(["Small", "Medium"])

    num_features = random.randint(3, 20)

    num_records = (
        random.randint(300, 2000)
        if dataset_size == "Small"
        else random.randint(2000, 10000)
    )

    data.append(
        create_row(
            "Classification",
            dataset_size,
            num_features,
            num_records,
            random.choice(["Medium", "High"]),
            "High",
            "High",
            "Decision Tree"
        )
    )


# --------------------------------------------------
# 3. Logistic Regression
# --------------------------------------------------

for _ in range(30):

    dataset_size = random.choice(["Small", "Medium"])
    num_features = random.randint(3, 15)

    num_records = (
        random.randint(500, 3000)
        if dataset_size == "Small"
        else random.randint(3000, 12000)
    )

    data.append(
        create_row(
            "Classification",
            dataset_size,
            num_features,
            num_records,
            random.choice(["Medium", "High"]),
            "High",
            "High",
            "Logistic Regression"
        )
    )


# --------------------------------------------------
# 4. SVM
# --------------------------------------------------

for _ in range(30):

    dataset_size = random.choice(["Small", "Medium"])
    num_features = random.randint(10, 30)

    num_records = (
        random.randint(500, 3000)
        if dataset_size == "Small"
        else random.randint(3000, 12000)
    )

    data.append(
        create_row(
            "Classification",
            dataset_size,
            num_features,
            num_records,
            "High",
            random.choice(["Low", "Medium"]),
            "Low",
            "SVM"
        )
    )


# --------------------------------------------------
# 5. Naive Bayes
# --------------------------------------------------

for _ in range(30):

    dataset_size = "Small"
    num_features = random.randint(3, 15)
    num_records = random.randint(500, 5000)

    data.append(
        create_row(
            "Classification",
            dataset_size,
            num_features,
            num_records,
            random.choice(["Medium", "High"]),
            "High",
            "Medium",
            "Naive Bayes"
        )
    )


# --------------------------------------------------
# 6. KNN
# --------------------------------------------------

for _ in range(30):

    dataset_size = "Small"
    num_features = random.randint(2, 10)
    num_records = random.randint(300, 2000)

    data.append(
        create_row(
            "Classification",
            dataset_size,
            num_features,
            num_records,
            "Medium",
            "Medium",
            "High",
            "KNN"
        )
    )


# --------------------------------------------------
# 7. Linear Regression
# --------------------------------------------------

for _ in range(30):

    dataset_size = random.choice(["Small", "Medium"])

    num_features = random.randint(2, 15)

    num_records = (
        random.randint(300, 3000)
        if dataset_size == "Small"
        else random.randint(3000, 12000)
    )

    data.append(
        create_row(
            "Regression",
            dataset_size,
            num_features,
            num_records,
            random.choice(["Medium", "High"]),
            "High",
            "High",
            "Linear Regression"
        )
    )


# --------------------------------------------------
# 8. K-Means
# --------------------------------------------------

for _ in range(30):

    dataset_size = random.choice(["Medium", "Large"])

    num_features = random.randint(2, 30)

    num_records = (
        random.randint(3000, 15000)
        if dataset_size == "Medium"
        else random.randint(15000, 100000)
    )

    data.append(
        create_row(
            "Clustering",
            dataset_size,
            num_features,
            num_records,
            "Medium",
            random.choice(["Medium", "High"]),
            "Medium",
            "K-Means"
        )
    )


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

df.to_csv(
    "ai/training_data.csv",
    index=False
)


# --------------------------------------------------
# Display Information
# --------------------------------------------------

print("Dataset generated successfully.")

print("\nTotal records:")
print(len(df))

print("\nAlgorithm distribution:")
print(
    df["recommended_algorithm"].value_counts()
)

print("\nProblem type distribution:")
print(
    df["problem_type"].value_counts()
)

print("\nDataset saved to:")
print("ai/training_data.csv")