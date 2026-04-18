 import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("💼 Salary Predictor")
st.write("Estimate salary based on experience and skills")

st.write("----------------------------------------")

# Inputs
experience = st.slider("Years of Experience", 0, 10, 2)

skills = st.multiselect(
    "Select your skills",
    ["Python", "SQL", "Machine Learning", "Data Analysis", "Deep Learning", "NLP"]
)

skill_score = len(skills)

# Sample dataset (simple but realistic)
X = np.array([
    [0, 1],
    [1, 2],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6],
    [7, 6],
    [8, 7],
    [9, 8]
])

y = np.array([
    300000,
    400000,
    480000,
    600000,
    720000,
    850000,
    950000,
    1050000,
    1150000,
    1300000
])

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
if st.button("Predict Salary"):

    input_data = np.array([[experience, skill_score]])
    prediction = model.predict(input_data)[0]

    st.subheader("Estimated Salary")
    st.success(f"₹ {int(prediction):,}")

    st.write("----------------------------------------")

    # Graph
    st.subheader("📊 Experience vs Salary Trend")

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], y)
    ax.set_xlabel("Experience (Years)")
    ax.set_ylabel("Salary")
    st.pyplot(fig)

    st.write("----------------------------------------")

    # Model understanding
    st.subheader("🧠 What affects salary more?")

    coef_exp = model.coef_[0]
    coef_skill = model.coef_[1]

    st.write(f"Experience impact: {round(coef_exp, 2)}")
    st.write(f"Skills impact: {round(coef_skill, 2)}")

    st.write("----------------------------------------")

    # Suggestions
    if prediction < 500000:
        st.warning("Try building more skills to improve your salary.")
    elif prediction < 900000:
        st.info("Good progress, keep improving your profile.")
    else:
        st.success("Strong profile! You're in a great position.")
