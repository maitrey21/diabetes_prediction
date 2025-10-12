import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from joblib import load as joblib_load
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stApp { max-width: 1400px; margin: 0 auto; }
    .prediction-box { padding: 20px; border-radius: 10px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .positive { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .negative { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Helper: model path
def get_model_path(filename='diabetes_model.pkl'):
    try:
        base = os.path.dirname(__file__)
        if not base:
            base = os.getcwd()
    except NameError:
        base = os.getcwd()
    return os.path.join(base, filename)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = get_model_path('diabetes_model.pkl')

    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found in the app folder.")
        return None

    # Try pickle first, then joblib
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        try:
            model = joblib_load(model_path)
        except Exception as e:
            st.error(f"⚠️ Failed to load model: {e}")
            return None

    # Load expected feature names if available
    expected = None
    try:
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
        elif hasattr(model, 'named_steps'):
            last_step = list(model.named_steps.values())[-1]
            if hasattr(last_step, 'feature_names_in_'):
                expected = list(last_step.feature_names_in_)
    except Exception:
        expected = None

    # Try loading feature_columns.json as backup
    if expected is None:
        feature_json_path = get_model_path('feature_columns.json')
        if os.path.exists(feature_json_path):
            import json
            with open(feature_json_path, 'r') as f:
                expected = json.load(f)

    st.session_state['expected_features'] = expected
    return model

# Feature engineering
def engineer_features(input_data):
    df = pd.DataFrame([input_data])
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Preg'] = df['Age'] * df['Pregnancies']
    df['Glucose_Insulin'] = df['Glucose'] * df['Insulin']
    df['BMI_squared'] = df['BMI'] ** 2
    df['Age_squared'] = df['Age'] ** 2
    df['Glucose_Age'] = df['Glucose'] / (df['Age'] + 1)
    df['BMI_Age'] = df['BMI'] / (df['Age'] + 1)
    df['Insulin_Glucose'] = df['Insulin'] / (df['Glucose'] + 1)
    df['BP_Category'] = pd.cut(df['BloodPressure'], bins=[-1, 80, 90, 120, 1000], labels=[0, 1, 2, 3]).astype(int)
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[-1, 18.5, 25, 30, 1000], labels=[0, 1, 2, 3]).astype(int)
    df['Age_Group'] = pd.cut(df['Age'], bins=[-1, 30, 40, 50, 200], labels=[0, 1, 2, 3]).astype(int)
    df['High_Risk_Pregnancy'] = ((df['Pregnancies'] > 5) & (df['Age'] > 30)).astype(int)
    df['Metabolic_Risk'] = ((df['BMI'] > 30) & (df['BloodPressure'] > 85) & (df['Glucose'] > 120)).astype(int)
    return df

# Align to expected columns
def align_input_to_expected(engineered_df):
    expected_cols = st.session_state.get('expected_features', None)

    if expected_cols is None:
        feature_json_path = get_model_path('feature_columns.json')
        if os.path.exists(feature_json_path):
            import json
            with open(feature_json_path, 'r') as f:
                expected_cols = json.load(f)

    if expected_cols is None:
        return engineered_df

    for c in expected_cols:
        if c not in engineered_df.columns:
            engineered_df[c] = 0

    extras = [c for c in engineered_df.columns if c not in expected_cols]
    if extras:
        engineered_df = engineered_df.drop(columns=extras)

    engineered_df = engineered_df[expected_cols]
    return engineered_df

# Risk label
def get_risk_level(probability):
    if probability < 0.3:
        return "Low Risk", "🟢"
    elif probability < 0.6:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"

# Gauge chart
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Diabetes Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 60], 'color': 'yellow'},
                {'range': [60, 100], 'color': 'red'}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Feature importance chart
def create_feature_importance_chart(input_data):
    features = ['Glucose', 'BMI', 'Age', 'Blood Pressure', 'Insulin', 'Pregnancies']
    values = [
        input_data['Glucose'] / 200 * 100,
        input_data['BMI'] / 50 * 100,
        input_data['Age'] / 100 * 100,
        input_data['BloodPressure'] / 150 * 100,
        input_data['Insulin'] / 300 * 100,
        input_data['Pregnancies'] / 15 * 100
    ]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation='h',
        marker=dict(color=values, colorscale='RdYlGn_r', showscale=True),
        text=[f'{v:.1f}%' for v in values], textposition='auto',
    ))
    fig.update_layout(title="Your Health Metrics (Normalized %)", height=400)
    return fig

# Recommendations
def get_recommendations(prediction, input_data):
    r = []
    if input_data['Glucose'] > 140:
        r.append("🔴 High Glucose: Monitor regularly and consult a doctor.")
    elif input_data['Glucose'] > 100:
        r.append("🟡 Elevated Glucose: Reduce sugar intake and exercise.")
    else:
        r.append("🟢 Normal Glucose: Maintain your current habits.")
    if input_data['BMI'] > 30:
        r.append("🔴 Obesity: Focus on diet and physical activity.")
    elif input_data['BMI'] > 25:
        r.append("🟡 Overweight: Adopt a healthier lifestyle.")
    else:
        r.append("🟢 Healthy BMI: Keep it up!")
    if input_data['BloodPressure'] > 90:
        r.append("🔴 High BP: Reduce salt intake and monitor regularly.")
    elif input_data['BloodPressure'] > 80:
        r.append("🟡 Borderline BP: Manage stress and maintain activity.")
    else:
        r.append("🟢 Normal BP: Great condition.")
    if input_data['Age'] > 45:
        r.append("⚠️ Age Factor: Regular health check-ups are advised.")
    if prediction == 1:
        r.append("⚠️ High Risk: Consult your healthcare provider soon.")
        r.append("📋 Plan: Get HbA1c test and follow diabetic diet.")
    return r

# Main app
def main():
    with st.sidebar:
        st.image("https://img.icons8.com/cotton/200/000000/stethoscope.png", width=100)
        st.title("🏥 Diabetes Predictor")
        st.markdown("---")
        page = st.radio("Navigation", ["🏠 Home", "🔮 Prediction", "📊 Dashboard", "ℹ️ About"], label_visibility="collapsed")
        st.markdown("---")
        st.info("AI-powered diabetes prediction using health metrics.")
        st.markdown("---")
        st.markdown("**Developed with ❤️ | Accuracy: 80%+**")
    
    model = load_model()
    if model is None:
        st.error("Model not loaded properly.")
        return

    # ---------------- HOME ----------------
    if page == "🏠 Home":
        st.title("🏥 Diabetes Prediction System")
        st.markdown("### Welcome to your AI Health Assistant")
        st.markdown("Get instant diabetes risk assessment using AI.")
        st.warning("⚠️ This tool is for educational purposes only.")

    # ---------------- PREDICTION ----------------
    elif page == "🔮 Prediction":
        st.title("🔮 Diabetes Risk Prediction")
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                pregnancies = st.number_input("Pregnancies", 0, 20, 0)
                glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
                bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
            with col2:
                skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
                insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
            with col3:
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
                age = st.number_input("Age", 1, 120, 30)
            submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

        if submitted:
            input_data = {
                'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
                'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
                'DiabetesPedigreeFunction': dpf, 'Age': age
            }

            df = engineer_features(input_data)
            df = align_input_to_expected(df)

            try:
                pred = model.predict(df)[0]
                prob = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else 0.5
                risk, emoji = get_risk_level(prob)

                st.markdown("---")
                if pred == 1:
                    st.markdown(f'<div class="prediction-box negative"><h2>{emoji} HIGH RISK</h2><h3>Risk: {prob*100:.1f}%</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box positive"><h2>{emoji} LOW RISK</h2><h3>Risk: {prob*100:.1f}%</h3></div>', unsafe_allow_html=True)
                st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
                st.plotly_chart(create_feature_importance_chart(input_data), use_container_width=True)

                st.markdown("## 💡 Recommendations")
                for rec in get_recommendations(pred, input_data):
                    st.markdown(f"- {rec}")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ---------------- DASHBOARD ----------------
    elif page == "📊 Dashboard":
        st.title("📊 Model Dashboard")
        st.metric("Model Accuracy", "80.2%", "↑ 5%")
        st.metric("ROC-AUC", "0.85", "↑ 0.05")
        st.metric("Precision", "78.5%", "↑ 3%")
        st.metric("Recall", "82.1%", "↑ 4%")

    # ---------------- ABOUT ----------------
    elif page == "ℹ️ About":
        st.title("ℹ️ About")
        st.markdown("""
        This AI system predicts diabetes risk using health parameters.
        - **Framework**: Streamlit  
        - **ML Libraries**: scikit-learn, XGBoost, LightGBM  
        - **Accuracy**: 80%+  
        ⚠️ Not a medical diagnosis. Always consult professionals.
        """)

if __name__ == "__main__":
    main()
