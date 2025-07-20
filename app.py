import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

# Title
st.title("🍷 Wine Quality Prediction App")
st.markdown("Built using **Random Forest Classifier** on the UCI Red Wine dataset.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    return df

data = load_data()

# Tabs
tabs = st.tabs(["📊 Data Exploration", "📈 Model & Accuracy", "🔮 Predict Wine Quality"])

# =======================================
# Tab 1: EDA
# =======================================
with tabs[0]:
    st.subheader("📊 Exploratory Data Analysis")
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Data Shape")
    st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

    st.write("### Null Values")
    st.write(data.isnull().sum())

    st.write("### Statistical Summary")
    st.dataframe(data.describe())

    st.write("### Wine Quality Count")
    fig1 = sns.countplot(x='quality', data=data)
    st.pyplot(fig1.figure)

    st.write("### Volatile Acidity vs Quality")
    fig2 = plt.figure(figsize=(6,4))
    sns.barplot(x='quality', y='volatile acidity', data=data)
    st.pyplot(fig2)

    st.write("### Citric Acid vs Quality")
    fig3 = plt.figure(figsize=(6,4))
    sns.barplot(x='quality', y='citric acid', data=data)
    st.pyplot(fig3)

    st.write("### Correlation Heatmap")
    corr = data.corr()
    fig4 = plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.1f', cmap='Blues')
    st.pyplot(fig4)

# =======================================
# Tab 2: Model Training
# =======================================
with tabs[1]:
    st.subheader("⚙️ Model Training & Accuracy")
    x = data.drop("quality", axis=1)
    y = data["quality"].apply(lambda val: 1 if val >= 7 else 0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model trained using Random Forest")
    st.metric(label="Test Accuracy", value=f"{accuracy*100:.2f}%")

# =======================================
# Tab 3: Prediction
# =======================================
with tabs[2]:
    st.subheader("🔮 Predict Wine Quality")

    st.markdown("Input the wine's chemical characteristics below:")

    col1, col2, col3 = st.columns(3)

    fixed_acidity = col1.slider("Fixed Acidity", 4.0, 16.0, 7.3)
    volatile_acidity = col2.slider("Volatile Acidity", 0.1, 1.5, 0.65)
    citric_acid = col3.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = col1.slider("Residual Sugar", 0.5, 15.0, 1.2)
    chlorides = col2.slider("Chlorides", 0.01, 0.2, 0.065)
    free_sulfur_dioxide = col3.slider("Free Sulfur Dioxide", 1.0, 75.0, 15.0)
    total_sulfur_dioxide = col1.slider("Total Sulfur Dioxide", 6.0, 300.0, 21.0)
    density = col2.slider("Density", 0.990, 1.005, 0.9946)
    pH = col3.slider("pH", 2.5, 4.5, 3.39)
    sulphates = col1.slider("Sulphates", 0.3, 2.0, 0.47)
    alcohol = col2.slider("Alcohol", 8.0, 15.0, 10.0)

    if st.button("Predict Quality"):
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ]])
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("✅ It is a **Good Quality Wine**!")
        else:
            st.error("❌ It is a **Bad Quality Wine**.")
