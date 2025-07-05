import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ Set page config at the very top (safe position)
st.set_page_config(page_title="Credit Scoring Predictor", page_icon="💳")

# Load and preprocess the dataset
@st.cache_data
def load_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
    df.rename(columns={'Survived': 'Creditworthy'}, inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)

    X = df.drop('Creditworthy', axis=1)
    y = df['Creditworthy']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# Load model and scaler
model, scaler = load_model()

# Streamlit UI
st.title("💳 Credit Scoring Predictor")
st.markdown("Enter the following details to check if the person is **creditworthy** or not.")

# User input fields
pclass = st.selectbox("Credit Class (Pclass)", [1, 2, 3])
age = st.slider("Age", 10, 80, 30)
sibsp = st.number_input("Number of Siblings/Spouse Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare / Income", 0.0, 600.0, 50.0)

# Predict button
if st.button("Predict Creditworthiness"):
    input_data = np.array([[pclass, age, sibsp, parch, fare]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ The person is predicted to be **CREDITWORTHY** (Confidence: {prediction_proba:.2f})")
    else:
        st.error(f"❌ The person is predicted to be **NOT CREDITWORTHY** (Confidence: {1 - prediction_proba:.2f})")

# Footer
st.markdown("---")
st.caption("Developed by Syed Muzammil Hussain | Internship Project - CodeAlpha")
