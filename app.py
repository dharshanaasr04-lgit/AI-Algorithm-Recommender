import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Algorithm Recommender",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 35px;
    }

    .recommendation-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #e8f8ee;
        border: 1px solid #b7e4c7;
        font-size: 20px;
        font-weight: 600;
    }

    .confidence {
        font-size: 30px;
        font-weight: 600;
    }

    .description-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #e8f1ff;
        border: 1px solid #c7dbff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_data = joblib.load(
        "ai/algorithm_recommender.pkl"
    )

    model = model_data["model"]
    encoder = model_data["encoder"]
    feature_columns = model_data["feature_columns"]

    return model, encoder, feature_columns


try:

    model, encoder, feature_columns = load_model()

except Exception as e:

    st.error("Unable to load the trained AI model.")

    st.code(str(e))

    st.stop()


# ============================================================
# ALGORITHM INFORMATION
# ============================================================

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


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About")

    st.write(
        "This application uses a trained machine learning "
        "model to recommend a suitable algorithm based on "
        "your project requirements."
    )

    st.divider()

    st.header("🤖 Algorithms")

    st.write("• Random Forest")
    st.write("• Decision Tree")
    st.write("• Logistic Regression")
    st.write("• SVM")
    st.write("• Naive Bayes")
    st.write("• KNN")
    st.write("• Linear Regression")
    st.write("• K-Means")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Algorithm Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Algorithm Selection System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PROJECT DETAILS
# ============================================================

st.header("📊 Project Details")


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    problem_type = st.selectbox(
        "Problem Type",
        [
            "Classification",
            "Regression",
            "Clustering"
        ]
    )

    dataset_size = st.selectbox(
        "Dataset Size",
        [
            "Small",
            "Medium",
            "Large"
        ]
    )

    num_features = st.selectbox(
        "Number of Features",
        [
            3,
            5,
            8,
            10,
            15,
            20,
            25,
            30,
            40,
            50
        ],
        index=3
    )

    num_records = st.selectbox(
        "Number of Records",
        [
            500,
            1000,
            5000,
            10000,
            20000,
            50000,
            100000
        ],
        index=1
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    accuracy_priority = st.selectbox(
        "Accuracy Priority",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    speed_priority = st.selectbox(
        "Speed Priority",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    interpretability = st.selectbox(
        "Interpretability",
        [
            "Low",
            "Medium",
            "High"
        ]
    )


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

st.write("")

recommend_button = st.button(
    "🔍 Recommend Algorithm",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if recommend_button:

    # --------------------------------------------------------
    # Create user input DataFrame
    # --------------------------------------------------------

    user_data = pd.DataFrame({

        "problem_type": [problem_type],

        "dataset_size": [dataset_size],

        "num_features": [num_features],

        "num_records": [num_records],

        "accuracy_priority": [accuracy_priority],

        "speed_priority": [speed_priority],

        "interpretability": [interpretability]

    })


    # --------------------------------------------------------
    # Categorical and numerical columns
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    try:

        encoded_data = encoder.transform(
            user_data[categorical_columns]
        )

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(
                categorical_columns
            )
        )


        # ----------------------------------------------------
        # Combine numerical + encoded features
        # ----------------------------------------------------

        processed_input = pd.concat(
            [
                user_data[numerical_columns]
                .reset_index(drop=True),

                encoded_df
                .reset_index(drop=True)
            ],
            axis=1
        )


        # ----------------------------------------------------
        # Match training feature order
        # ----------------------------------------------------

        processed_input = processed_input.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # Predict algorithm
        # ----------------------------------------------------

        prediction = model.predict(
            processed_input
        )

        recommended_algorithm = prediction[0]


        # ----------------------------------------------------
        # Calculate confidence
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                processed_input
            )[0]

            confidence = max(probabilities) * 100

        else:

            confidence = 0


        # ----------------------------------------------------
        # Get description
        # ----------------------------------------------------

        description = algorithm_info.get(
            recommended_algorithm,
            {
                "description":
                "This algorithm is recommended based "
                "on the characteristics of your project."
            }
        )["description"]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.header("🎯 AI Recommendation")


        # ----------------------------------------------------
        # Recommended Algorithm
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="recommendation-box">
                Recommended Algorithm: {recommended_algorithm}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.subheader("Model Confidence")

        st.markdown(
            f'<div class="confidence">{confidence:.2f}%</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Why recommendation?
        # ----------------------------------------------------

        st.subheader("💡 Why this recommendation?")

        st.write(
            f"✓ Problem Type: **{problem_type}**"
        )

        st.write(
            f"✓ Dataset Size: **{dataset_size}**"
        )

        st.write(
            f"✓ Number of Features: **{num_features}**"
        )

        st.write(
            f"✓ Number of Records: **{num_records}**"
        )

        st.write(
            f"✓ Accuracy Priority: **{accuracy_priority}**"
        )

        st.write(
            f"✓ Speed Priority: **{speed_priority}**"
        )

        st.write(
            f"✓ Interpretability: **{interpretability}**"
        )


        # ----------------------------------------------------
        # Algorithm Description
        # ----------------------------------------------------

        st.subheader("📚 Algorithm Description")

        st.markdown(
            f"""
            <div class="description-box">
                {description}
            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            "An error occurred while generating "
            "the recommendation."
        )

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Algorithm Recommender | "
    "Machine Learning Project"
)