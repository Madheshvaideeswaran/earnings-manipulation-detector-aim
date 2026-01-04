import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler  # ✅ Corrected import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# 💛🖤 Custom Yellow & Black Theme 💛🖤
st.markdown("""
<style>
    :root {
        --primary-bg: #000000;
        --primary-text: #ffff00;
        --secondary-bg: #1a1a1a;
        --accent: #ffff00;
        --card-bg: #222222;
    }
    .css-18e3th9 {
        padding: 2rem 1rem 10rem;
        background-color: var(--primary-bg);
        color: var(--primary-text);
    }
    .css-1d391kg {
        background-color: var(--primary-bg);
    }
    .css-1offfwp, .css-1v0mbdj, .css-1q8dd3e {
        background-color: var(--card-bg);
        color: var(--primary-text);
        border: 1px solid var(--accent);
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown {
        color: var(--accent) !important;
    }
    .stButton>button {
        background-color: var(--accent);
        color: var(--primary-bg);
        border: 2px solid var(--accent);
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: var(--primary-bg);
        color: var(--accent);
    }
    .stMetric {
        background-color: var(--card-bg);
        border: 1px solid var(--accent);
        padding: 1rem;
        border-radius: 8px;
    }
    .stMetric .css-10trblm {
        color: var(--accent) !important;
    }
    .stMetric .css-1offfwp {
        color: var(--accent) !important;
    }
    .stSelectbox, .stSlider, .stCheckbox, .stFileUploader {
        color: var(--primary-text);
    }
    .stFileUploader .css-1m4d17d {
        color: var(--primary-text);
    }
    .stSidebar {
        background-color: var(--secondary-bg) !important;
        color: var(--accent) !important;
    }
    .stSidebar .css-1d391kg {
        background-color: var(--secondary-bg) !important;
    }
    .stSidebar * {
        color: var(--accent) !important;
    }
    input, textarea, select {
        background-color: var(--card-bg) !important;
        color: var(--accent) !important;
        border: 1px solid var(--accent) !important;
    }
    .stCodeBlock {
        background-color: var(--card-bg);
        color: var(--accent);
        border: 1px solid var(--accent);
    }
    .stDataFrame {
        color: var(--primary-text);
    }
    table {
        color: var(--accent) !important;
    }
    thead tr th {
        background-color: var(--card-bg) !important;
        color: var(--accent) !important;
    }
    tbody tr td {
        background-color: var(--primary-bg) !important;
        color: var(--accent) !important;
    }
</style>
""", unsafe_allow_html=True)

# Evaluation function
def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

# Parameter grids for tuning
PARAM_GRIDS = {
    "SVM": {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    "Naive Bayes": {
        'var_smoothing': np.logspace(0, -9, num=10)
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.5]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
}

# Helper: Does model need scaled features?
def needs_scaling(model_name):
    return model_name in ["SVM", "KNN"]

# Model factory
def get_model(model_name):
    if model_name == "SVM":
        return SVC(probability=True, random_state=42)
    elif model_name == "KNN":
        return KNeighborsClassifier()
    elif model_name == "Naive Bayes":
        return GaussianNB()
    elif model_name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)
    elif model_name == "XGBoost":
        return XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

# Streamlit UI
st.set_page_config(page_title="Earnings Manipulation Detector", layout="wide")
st.title("🔍 Earnings Manipulation Classification Dashboard")
st.markdown("""
Upload your financial dataset and compare machine learning models to detect earnings manipulation.
Ensure your Excel file includes these columns:  
`DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `ACCR`, `LEVI`, `Manipulator` (with values 'Yes'/'No').
""")

# File uploader
uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI','Manipulator']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Missing required columns. Expected: {required_cols}")
            st.stop()

        # Encode target
        y = df['Manipulator'].map({'No': 0, 'Yes': 1})
        if y.isnull().any():
            st.error("❌ 'Manipulator' column must contain only 'Yes' or 'No'.")
            st.stop()
        X = df[required_cols[:-1]]

        # Sidebar controls
        st.sidebar.header("⚙️ Model Configuration")
        model_name = st.sidebar.selectbox(
            "Select Model",
            ["SVM", "KNN", "Naive Bayes", "AdaBoost", "XGBoost"]
        )
        test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.25, step=0.05)
        use_tuning = st.sidebar.checkbox("🔍 Enable Hyperparameter Tuning (GridSearchCV)", value=False)
        run_button = st.sidebar.button("🚀 Run Model")

        if run_button:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Scale if needed
            if needs_scaling(model_name):
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train)
                X_test_pred = scaler.transform(X_test)
            else:
                X_train_fit, X_test_pred = X_train, X_test

            # Get model
            model = get_model(model_name)

            # Train (with or without tuning)
            if use_tuning:
                with st.spinner(f"🔬 Tuning {model_name} with GridSearchCV (may take 1-2 minutes)..."):
                    grid = GridSearchCV(
                        model,
                        PARAM_GRIDS[model_name],
                        cv=5,
                        scoring='roc_auc',
                        n_jobs=-1,
                        refit=True
                    )
                    grid.fit(X_train_fit, y_train)
                    best_model = grid.best_estimator_
                    best_params = grid.best_params_
            else:
                with st.spinner(f"🏋️ Training {model_name} with default parameters..."):
                    model.fit(X_train_fit, y_train)
                    best_model = model
                    best_params = "Default (no tuning)"

            # Predict
            y_pred = best_model.predict(X_test_pred)
            y_prob = best_model.predict_proba(X_test_pred)[:, 1]

            # Evaluate
            metrics = evaluate(y_test, y_pred, y_prob)

            # Results section
            st.subheader(f"📈 Performance: {model_name}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("Precision", f"{metrics['Precision']:.4f}")
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            with col2:
                st.metric("F1-score", f"{metrics['F1-score']:.4f}")
                st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

            st.markdown("### 🔧 Best Parameters")
            if isinstance(best_params, dict):
                st.json(best_params)
            else:
                st.code(best_params)

            # Beneish baseline
            st.markdown("---")
            st.subheader("📊 Beneish M-Score Rule (Static Baseline)")
            beneish = {
                "Accuracy": 0.8364,
                "Precision": 0.5556,
                "Recall": 0.5000,
                "F1-score": 0.5263,
                "ROC-AUC": 0.9044
            }
            st.dataframe(pd.DataFrame([beneish]).round(4))

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload an Excel file to begin analysis.")