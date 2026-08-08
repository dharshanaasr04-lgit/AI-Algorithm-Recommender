import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Algorithm Recommender",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_data = joblib.load(
        "ai/algorithm_recommender.pkl"
    )

    return (
        model_data["model"],
        model_data["encoder"],
        model_data["feature_columns"]
    )


model, encoder, feature_columns = load_model()


# ============================================================
# 3. ALGORITHM INFORMATION
# ============================================================

algorithm_info = {

    "Random Forest": {
        "description":
            "An ensemble learning algorithm that combines "
            "multiple decision trees. It is useful for complex "
            "classification and regression datasets.",

        "type": "Supervised Learning",

        "best_for":
            "Large and complex datasets with many features."
    },

    "Decision Tree": {
        "description":
            "A tree-based algorithm that makes decisions using "
            "a sequence of simple rules.",

        "type": "Supervised Learning",

        "best_for":
            "Problems where interpretability is important."
    },

    "Logistic Regression": {
        "description":
            "A simple and efficient classification algorithm "
            "used to predict categorical outcomes.",

        "type": "Supervised Learning",

        "best_for":
            "Simple classification problems."
    },

    "SVM": {
        "description":
            "Support Vector Machine finds an effective decision "
            "boundary between classes.",

        "type": "Supervised Learning",

        "best_for":
            "High-dimensional classification problems."
    },

    "Naive Bayes": {
        "description":
            "A probabilistic algorithm based on Bayes' theorem. "
            "It is fast and works well with smaller datasets.",

        "type": "Supervised Learning",

        "best_for":
            "Text classification and smaller datasets."
    },

    "KNN": {
        "description":
            "K-Nearest Neighbors predicts a result based on "
            "similar nearby data points.",

        "type": "Supervised Learning",

        "best_for":
            "Small datasets where similar examples are useful."
    },

    "Linear Regression": {
        "description":
            "A regression algorithm used to predict numerical "
            "values based on relationships between variables.",

        "type": "Supervised Learning",

        "best_for":
            "Numerical prediction and regression problems."
    },

    "K-Means": {
        "description":
            "An unsupervised clustering algorithm that groups "
            "similar data points into clusters.",

        "type": "Unsupervised Learning",

        "best_for":
            "Finding natural groups in unlabeled data."
    }
}


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .recommendation-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 20px;
        text-align: center;
    }

    .algorithm-name {
        font-size: 32px;
        font-weight: 700;
        margin: 10px;
    }

    .confidence {
        font-size: 22px;
        font-weight: 600;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HEADER
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
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About")

    st.write(
        "This application uses a trained machine learning "
        "model to recommend a suitable algorithm based on "
        "your project requirements."
    )

    st.divider()

    st.subheader("🤖 Algorithms")

    st.write("• Random Forest")
    st.write("• Decision Tree")
    st.write("• Logistic Regression")
    st.write("• SVM")
    st.write("• Naive Bayes")
    st.write("• KNN")
    st.write("• Linear Regression")
    st.write("• K-Means")

    st.divider()

    st.caption(
        "AI Algorithm Recommender\n\n"
        "B.Sc. Information Technology Project"
    )


# ============================================================
# 7. PROJECT DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Project Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


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

    num_features = st.number_input(
        "Number of Features",
        min_value=1,
        value=10,
        step=1
    )

    num_records = st.number_input(
        "Number of Records",
        min_value=1,
        value=1000,
        step=1
    )


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


st.write("")


# ============================================================
# 8. RECOMMENDATION BUTTON
# ============================================================

recommend_button = st.button(
    "🔍 Recommend Algorithm",
    use_container_width=True
)


# ============================================================
# 9. AI PREDICTION
# ============================================================

if recommend_button:

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Feature columns
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

    encoded_data = encoder.transform(
        user_data[categorical_columns]
    )


    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )


    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    processed_input = pd.concat(
        [
            user_data[numerical_columns].reset_index(
                drop=True
            ),

            encoded_df.reset_index(
                drop=True
            )
        ],
        axis=1
    )


    # --------------------------------------------------------
    # Match training feature order
    # --------------------------------------------------------

    processed_input = processed_input.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = model.predict(
        processed_input
    )

    recommended_algorithm = prediction[0]


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        processed_input
    )[0]

    confidence = max(probabilities) * 100


    # --------------------------------------------------------
    # Algorithm details
    # --------------------------------------------------------

    info = algorithm_info[
        recommended_algorithm
    ]


    # ========================================================
    # 10. RESULT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 AI Recommendation</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="recommendation-card">

            <div>Recommended Algorithm</div>

            <div class="algorithm-name">
                {recommended_algorithm}
            </div>

            <div class="confidence">
                Model Confidence: {confidence:.2f}%
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Confidence progress
    # --------------------------------------------------------

    st.write("")

    st.progress(
        min(int(confidence), 100)
    )


    # ========================================================
    # 11. WHY THIS RECOMMENDATION?
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 Why this recommendation?</div>',
        unsafe_allow_html=True
    )


    reason_col1, reason_col2 = st.columns(2)


    with reason_col1:

        st.write(
            f"✓ **Problem Type:** {problem_type}"
        )

        st.write(
            f"✓ **Dataset Size:** {dataset_size}"
        )

        st.write(
            f"✓ **Number of Features:** {num_features}"
        )

        st.write(
            f"✓ **Number of Records:** {num_records}"
        )


    with reason_col2:

        st.write(
            f"✓ **Accuracy Priority:** {accuracy_priority}"
        )

        st.write(
            f"✓ **Speed Priority:** {speed_priority}"
        )

        st.write(
            f"✓ **Interpretability:** {interpretability}"
        )

        st.write(
            f"✓ **Learning Type:** {info['type']}"
        )


    # ========================================================
    # 12. ALGORITHM DESCRIPTION
    # ========================================================

    st.markdown(
        '<div class="section-title">📚 Algorithm Description</div>',
        unsafe_allow_html=True
    )


    st.info(
        info["description"]
    )


    st.markdown(
        f"**Best suited for:** {info['best_for']}"
    )


# ============================================================
# 13. ALGORITHM GUIDE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📖 Algorithm Guide</div>',
    unsafe_allow_html=True
)

st.write(
    "Learn about the algorithms supported by this system."
)


for algorithm, info in algorithm_info.items():

    with st.expander(
        f"🤖 {algorithm}"
    ):

        st.write(
            f"**Type:** {info['type']}"
        )

        st.write(
            f"**Description:** {info['description']}"
        )

        st.write(
            f"**Best suited for:** {info['best_for']}"
        )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Algorithm Recommender | "
    "Machine Learning Project | "
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)