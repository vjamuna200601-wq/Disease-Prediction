import os
import joblib
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(
    page_title="Disease / Heart Risk Predictor",
    page_icon="🩺",
    layout="centered"
)


CANDIDATE_PATHS = [
    "heart_disease_risk_dataset_earlymed.csv",
    "healthcare_dataset.csv",
    "data/heart_disease_risk_dataset_earlymed.csv",
    "data/healthcare_dataset.csv",
]
ENCODER_PATH = "label_encoder.joblib"
COLUMNS_PATH = "feature_columns.joblib"
ENSEMBLE_PATH = "ensemble_model.joblib"


st.sidebar.title("ℹ️ About")
st.sidebar.write(
    "Predicts likely **disease** (multi-class) or **heart risk** (binary) "
    "from features in your CSV using an ensemble (GaussianNB + Logistic + RandomForest)."
)
uploaded_file = st.sidebar.file_uploader("Upload CSV (if not on disk)", type=["csv"])


@st.cache_data
def load_csv_from_disk(candidates):
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p), p
    return None, None

@st.cache_data
def load_uploaded(uploaded):
    return pd.read_csv(uploaded)

@st.cache_resource
def train_or_load_models(X, y):
    """Train ensemble or load cached model if available."""
    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y)

    if os.path.exists(ENSEMBLE_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(COLUMNS_PATH):
        try:
            ensemble = joblib.load(ENSEMBLE_PATH)
            saved_cols = joblib.load(COLUMNS_PATH)
            saved_encoder = joblib.load(ENCODER_PATH)

            if list(X.columns) == list(saved_cols) and set(saved_encoder.classes_) == set(label_enc.classes_):
                y_for_acc = saved_encoder.transform(y)
                acc = accuracy_score(y_for_acc, ensemble.predict(X))
                return ensemble, saved_encoder, saved_cols, acc
        except Exception:
            pass

    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    nb = ("nb", GaussianNB())
    log = ("log", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]))
    rf = ("rf", RandomForestClassifier(n_estimators=200, random_state=42))

    ensemble = VotingClassifier(estimators=[nb, log, rf], voting="soft")
    ensemble.fit(X_train, y_train)

   
    joblib.dump(ensemble, ENSEMBLE_PATH)
    joblib.dump(label_enc, ENCODER_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)

    acc = accuracy_score(y_test, ensemble.predict(X_test))
    return ensemble, label_enc, list(X.columns), acc

def is_binary_col(series: pd.Series) -> bool:
    vals = set(series.dropna().unique().tolist())
    return vals.issubset({0, 1}) and len(vals) <= 2


data, used_path = load_csv_from_disk(CANDIDATE_PATHS)
if uploaded_file is not None:
    data = load_uploaded(uploaded_file)
    used_path = "(uploaded)"

if data is None:
    st.error(
        "❌ Could not find a CSV on disk and nothing was uploaded.\n\n"
        "Fix it by either:\n"
        "1) Put your CSV next to app.py (or in data/) with one of these names:\n"
        "   - heart_disease_risk_dataset_earlymed.csv\n"
        "   - healthcare_dataset.csv\n"
        "2) Or upload your CSV using the **sidebar uploader**."
    )
    st.stop()

st.success(f"✅ Loaded dataset from {used_path}")
st.write(f"Rows: **{data.shape[0]}**, Columns: **{data.shape[1]}**")
with st.expander("Preview data"):
    st.dataframe(data.head())


suggest = None
for cand in ["disease", "Heart_Risk", "heart_risk", "target", "label", "outcome"]:
    if cand in data.columns:
        suggest = cand
        break

target_col = st.sidebar.selectbox(
    "Target column (what you want to predict)",
    options=list(data.columns),
    index=(list(data.columns).index(suggest) if suggest in data.columns else 0),
)
if target_col not in data.columns:
    st.error("The selected target column does not exist in the dataset.")
    st.stop()

X = data.drop(columns=[target_col])
y = data[target_col]


non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
if non_numeric:
    st.error(f"These non-numeric columns need encoding: {non_numeric}. Please convert before using.")
    st.stop()


with st.spinner("Training / loading model..."):
    ensemble, label_enc, feature_columns, accuracy = train_or_load_models(X, y)

with st.expander("📊 Model Info"):
    st.write("Classes:", list(label_enc.classes_))
    st.write("Ensemble accuracy (last split):", f"{accuracy:.2f}")

   
    y_pred_all = ensemble.predict(X)
    st.text("Classification Report:")
    st.text(classification_report(y, y_pred_all))

    cm = confusion_matrix(y, y_pred_all)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_enc.classes_, yticklabels=label_enc.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)


rf_model = ensemble.named_estimators_["rf"]
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_columns, "Importance": importances}).sort_values("Importance", ascending=False)
st.write("🔑 Feature Importance (RandomForest)")
st.bar_chart(feat_imp.set_index("Feature"))

st.info("👇 Enter feature values to predict. Binary features show as toggles; numeric as sliders/inputs.")


user_values = {}
cols = st.columns(2)
for i, colname in enumerate(feature_columns):
    col = cols[i % 2]
    series = X[colname]
    if is_binary_col(series):
        val = col.toggle(f"{colname}", value=False)
        user_values[colname] = 1 if val else 0
    else:
        mn, mx = float(series.min()), float(series.max())
        default = float(series.median())
        step = (mx - mn) / 100 if mx > mn else 1.0
        user_values[colname] = col.number_input(
            f"{colname}", value=default, min_value=mn, max_value=mx,
            step=step, format="%.4f"
        )

user_df = pd.DataFrame([user_values])

top_k = st.slider("Show top K predictions", 1, min(5, len(label_enc.classes_)), 3)
if st.button("🔍 Predict"):
    with st.spinner("Predicting..."):
        probs = ensemble.predict_proba(user_df)[0]
        labels = label_enc.inverse_transform(np.arange(len(probs)))
        prob_df = pd.DataFrame({"class": labels, "prob": probs}).sort_values("prob", ascending=False).reset_index(drop=True)

        st.subheader("📊 Top predictions")
        st.table(prob_df.head(top_k).style.format({"prob": "{:.2f}"}))

        st.subheader("📈 Confidence chart")
        st.bar_chart(prob_df.head(top_k).set_index("class")["prob"])

     
        csv = prob_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

        
        top_pred = str(prob_df.loc[0, "class"]).strip().lower()
        st.subheader("💡 Recommendation")

        if "flu" in top_pred:
            st.info("🌀 **Flu** → Rest, fluids, fever reducers; consult a doctor if symptoms worsen.")
        elif "typhoid" in top_pred:
            st.warning("⚠️ **Typhoid** → Seek medical care for diagnosis and antibiotics.")
        elif "cold" in top_pred:
            st.info("🤧 **Common Cold** → Rest, hydration, OTC meds if needed.")
        elif "risk" in target_col.lower() or set(label_enc.classes_) <= {0, 1} or set(map(str, label_enc.classes_)) <= {"0", "1"}:
            if top_pred in {"1", "true", "yes", "high", "high risk", "risk"} or ("high" in top_pred and "low" not in top_pred):
                st.error("❤️ **High cardiac risk predicted** → Please consult a cardiologist promptly.")
            else:
                st.success("✅ **Low cardiac risk predicted** → Maintain healthy lifestyle and regular checkups.")
        else:
            st.info("⚕️ If symptoms persist or worsen, consult a healthcare professional.")

      
        try:
            rf_model = ensemble.named_estimators_["rf"]
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(user_df)

            pred_class_index = np.argmax(probs)
            single_explanation = shap.Explanation(
                values=shap_values[pred_class_index][0],
                base_values=explainer.expected_value[pred_class_index],
                data=user_df.iloc[0].values,
                feature_names=user_df.columns.tolist()
            )

            st.subheader("🔍 Feature contribution (SHAP values)")
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.waterfall_plot(single_explanation, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"⚠️ SHAP explanation skipped: {e}")

