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

# Dataset
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

if st.button("Predict Salary"):

    input_data = np.array([[experience, skill_score]])
    prediction = model.predict(input_data)[0]

    st.subheader("Estimated Salary")
    st.success(f"₹ {int(prediction):,}")

    st.write("----------------------------------------")

    # 📊 Graph with regression line + predicted point
    st.subheader("📊 Experience vs Salary Trend")

    fig, ax = plt.subplots()

    # scatter actual data
    ax.scatter(X[:, 0], y, label="Data")

    # regression line
    x_line = np.linspace(0, 10, 100)
    y_line = model.predict(np.column_stack((x_line, np.full_like(x_line, skill_score))))
    ax.plot(x_line, y_line, linestyle="--", label="Model Prediction")

    # predicted point
    ax.scatter(experience, prediction, marker="x", s=100, label="Your Prediction")

    ax.set_xlabel("Experience (Years)")
    ax.set_ylabel("Salary")
    ax.legend()

    st.pyplot(fig)

    st.write("----------------------------------------")

    # 🧠 Model explanation
    st.subheader("🧠 What affects salary more?")

    coef_exp = model.coef_[0]
    coef_skill = model.coef_[1]

    st.write(f"Each year of experience increases salary by approx ₹{int(coef_exp):,}")
    st.write(f"Each additional skill increases salary by approx ₹{int(coef_skill):,}")

    st.write("----------------------------------------")

    # Suggestions
    if prediction < 500000:
        st.warning("Try building more skills to improve your salary.")
    elif prediction < 900000:
        st.info("Good progress, keep improving your profile.")
    else:
        st.success("Strong profile! You're in a great position.")