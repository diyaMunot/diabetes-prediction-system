import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from fpdf import FPDF
import plotly.graph_objects as go
import os
from datetime import datetime

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Diabetes Prediction AI", layout="wide", page_icon="🩺")

DATA_PATH = "diabetes.csv"
HISTORY_PATH = "history.csv"

PURPLE = "#7B61FF"
PINK = "#FF61C1"


# -----------------------------------------------------
# PDF REPORT FIXED (Name visible + Unicode safe)
# -----------------------------------------------------
def generate_pdf_report(patient_info: dict, risk_pct: float, model_name: str):

    # Clean Unicode (FPDF supports only ASCII)
    clean_info = {
        k: str(v).encode("ascii", "ignore").decode("ascii")
        for k, v in patient_info.items()
    }

    clean_model = model_name.encode("ascii", "ignore").decode("ascii")

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Diabetes Prediction Report", ln=True, align="C")

    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    # Patient Details Section
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Details:", ln=True)

    pdf.set_font("Arial", size=11)
    for k, v in clean_info.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)

    # Prediction Summary
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Prediction Summary:", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 7, f"Model Used: {clean_model}", ln=True)
    pdf.cell(0, 7, f"Estimated Risk: {risk_pct:.1f}%", ln=True)

    # Medical Advice
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Medical Advice:", ln=True)

    pdf.set_font("Arial", size=11)

    if risk_pct >= 50:
        pdf.multi_cell(0, 7,
                       "High diabetes risk. Please consult a medical doctor for further tests "
                       "(HbA1c, OGTT). Maintain diet and follow medical advice.")
    else:
        pdf.multi_cell(0, 7,
                       "Low diabetes risk. Maintain healthy diet, exercise and routine checkups.")

    return pdf.output(dest="S").encode("latin-1", "ignore")


# -----------------------------------------------------
# TRAINING FUNCTIONS
# -----------------------------------------------------
def load_data():
    return pd.read_csv(DATA_PATH)


def train_and_compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True)),
        ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "pipeline": model,
            "accuracy": accuracy_score(y_test, y_pred)
        }

    return results


# -----------------------------------------------------
# UI HEADER
# -----------------------------------------------------

st.title("🩺 Diabetes Prediction System")
st.caption("Machine Learning Based Diabetes Risk Prediction")

# -----------------------------------------------------
# LOAD DATA + TRAIN
# -----------------------------------------------------
try:
    data = load_data()
except:
    st.error("❌ diabetes.csv not found! Place it in the same folder as app.py.")
    st.stop()

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

with st.spinner("Training machine learning models..."):
    results = train_and_compare(X, y)

comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy (%)": [round(results[m]["accuracy"] * 100, 2) for m in results]
}).sort_values(by="Accuracy (%)", ascending=False)

st.markdown("### 🔬 Model Performance Comparison")
st.dataframe(comparison_df, height=180)

best_model_name = comparison_df.iloc[0]["Model"]

chosen_model_name = st.selectbox(
    "Choose Model for Prediction:",
    list(results.keys()),
    index=list(results.keys()).index(best_model_name)
)

model = results[chosen_model_name]["pipeline"]

# -----------------------------------------------------
# SIDEBAR INPUT SECTION
# -----------------------------------------------------
st.sidebar.header("Patient Information")

patient_name = st.sidebar.text_input("Patient Name", placeholder="Enter full name")

gender = st.sidebar.radio("Gender", ["Female", "Male", "Other"])

pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0, max_value=20, value=0 if gender != "Female" else 1
)

glucose = st.sidebar.number_input("Glucose", 0, 300, 120)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 25)

predict_btn = st.sidebar.button("Predict")


# -----------------------------------------------------
# PREDICTION HANDLER
# -----------------------------------------------------
if predict_btn:

    if patient_name.strip() == "":
        st.error("❌ Please enter the patient's name before predicting.")
        st.stop()

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin,
                            insulin, bmi, dpf, age]])

    prob = model.predict_proba(input_data)[0][1]
    risk_pct = prob * 100
    pred = int(model.predict(input_data)[0])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Prediction Result")
        if pred == 1:
            st.error(f"⚠️ High Diabetes Risk — {risk_pct:.1f}%")
        else:
            st.success(f"✅ Low Diabetes Risk — {risk_pct:.1f}%")

        st.write(f"**Model Used:** {chosen_model_name}")

    with col2:
        st.markdown("### 🎛 Risk Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": PINK}},
            title={"text": "Risk (%)"}
        ))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # SAVE HISTORY (Auto-Reset If Corrupted)
    # -----------------------------------------------------
    record = {
        "PatientName": patient_name,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Gender": gender,
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DPF": dpf,
        "Age": age,
        "RiskPct": round(risk_pct, 2),
        "Prediction": pred,
        "ModelUsed": chosen_model_name,
    }

    df = pd.DataFrame([record])

    # ✅ Auto-reset corrupted history file
    if os.path.exists(HISTORY_PATH):
        try:
            existing = pd.read_csv(HISTORY_PATH)
            existing = pd.concat([existing, df], ignore_index=True)
            existing.to_csv(HISTORY_PATH, index=False)

        except:
            # AUTO FIX
            df.to_csv(HISTORY_PATH, index=False)
    else:
        df.to_csv(HISTORY_PATH, index=False)

    # -----------------------------------------------------
    # PDF DOWNLOAD
    # -----------------------------------------------------
    pdf_bytes = generate_pdf_report(record, risk_pct, chosen_model_name)

    st.download_button(
        label="📥 Download Medical Report (PDF)",
        data=pdf_bytes,
        file_name=f"{patient_name}_Diabetes_Report.pdf",
        mime="application/pdf"
    )

# -----------------------------------------------------
# HISTORY DISPLAY
# -----------------------------------------------------
st.markdown("---")
st.markdown("### 🗃 Previous Predictions")

if os.path.exists(HISTORY_PATH):
    try:
        hist = pd.read_csv(HISTORY_PATH)
        st.dataframe(hist.sort_values("Timestamp", ascending=False), height=250)
    except:
        st.warning("History file was corrupted and auto-reset.")
else:
    st.info("No history yet. Make a prediction to get started!")


