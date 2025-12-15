import streamlit as st
import joblib as jb
import numpy as np

model = jb.load(r"C:\Users\mibra\OneDrive\Desktop\ML Project\online_gaming_behavior_dataset.pkl")

st.set_page_config(page_title="Player Churn Prediction", layout="centered")

st.title("Player Churn Risk Prediction App")
st.write("This app predicts **Churn Risk** using a Logistic Regression model.")

age = st.number_input("Age", min_value=10, max_value=100, value=25)
play_time = st.number_input("Play Time (Hours)", min_value=0.0, value=10.0)
in_game_purchases = st.number_input("In-Game Purchases", min_value=0.0, value=5.0)
sessions_per_week = st.number_input("Sessions Per Week", min_value=0, value=5)
avg_session_duration = st.number_input("Avg Session Duration (Minutes)", min_value=1.0, value=30.0)
player_level = st.number_input("Player Level", min_value=1, value=10)
achievements = st.number_input("Achievements Unlocked", min_value=0, value=3)

gender = st.selectbox("Gender", ["Male", "Female"])
location = st.selectbox("Location", ["Urban", "Rural"])
game_genre = st.selectbox("Game Genre", ["Action", "Adventure", "Puzzle", "Strategy"])
game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
engagement = st.selectbox("Engagement Level", ["Low", "Medium", "High"])

gender_male = 1 if gender == "Male" else 0
location_urban = 1 if location == "Urban" else 0

genre_action = 1 if game_genre == "Action" else 0
genre_adventure = 1 if game_genre == "Adventure" else 0
genre_puzzle = 1 if game_genre == "Puzzle" else 0

difficulty_medium = 1 if game_difficulty == "Medium" else 0
difficulty_hard = 1 if game_difficulty == "Hard" else 0

engagement_medium = 1 if engagement == "Medium" else 0
engagement_high = 1 if engagement == "High" else 0

if st.button("Predict Churn Risk"):
    features = np.array([[
        age, play_time, in_game_purchases, sessions_per_week, avg_session_duration, player_level, achievements, gender_male,
        location_urban, genre_action, genre_adventure, genre_puzzle, difficulty_medium, difficulty_hard,engagement_medium,
        engagement_high]])

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Churn Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Churn Risk ({probability*100:.2f}%)")

st.markdown("---")
st.caption("Logistic Regression model for player churn prediction")