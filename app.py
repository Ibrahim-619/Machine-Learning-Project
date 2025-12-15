import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and columns
model = joblib.load(r"C:\Users\mibra\OneDrive\Desktop\ML Project\churn.pkl")
scaler = joblib.load(r"C:\Users\mibra\OneDrive\Desktop\ML Project\scaler.pkl")
model_columns = joblib.load(r"C:\Users\mibra\OneDrive\Desktop\ML Project\model_columns.pkl")

# Function to predict churn
def predict_churn(
    age,
    gender,
    location,
    game_genre,
    playtime,
    purchases,
    game_difficulty,
    sessions,
    avg_session,
    player_level,
    achievements
):
    # Create input dataframe
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Location": location,
        "GameGenre": game_genre,
        "PlayTimeHours": playtime,
        "InGamePurchases": purchases,
        "GameDifficulty": game_difficulty,
        "SessionsPerWeek": sessions,
        "AvgSessionDurationMinutes": avg_session,
        "PlayerLevel": player_level,
        "AchievementsUnlocked": achievements
    }])

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_df)

    # Reindex to match model columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale numerical columns
    numerical_cols = [col for col in input_df.select_dtypes(include=np.number).columns if col in input_encoded.columns]
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

    # Predict churn probability
    churn_prob = model.predict_proba(input_encoded)[0][1]

    # Determine retention action
    if churn_prob > 0.7:
        action = "High Risk: Offer discounts or free rewards"
    elif churn_prob > 0.4:
        action = "Medium Risk: Personalized missions & reminders"
    else:
        action = "Low Risk: Loyalty rewards"

    return round(churn_prob, 3), action

# Streamlit App
st.title("🎮 Player Churn Risk Prediction")
st.write("Logistic Regression model based on engagement behavior.")

# Inputs
age = st.number_input("Age", min_value=0, value=25, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
location = st.selectbox("Location", ["North America", "Europe", "Asia", "Other"])
game_genre = st.selectbox("Game Genre", ["Action", "Strategy", "RPG", "Sports"])
playtime = st.number_input("Play Time (Hours)", min_value=0.0, value=10.0, step=1.0)
purchases = st.number_input("In-Game Purchases", min_value=0, value=0, step=1)
game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
sessions = st.number_input("Sessions Per Week", min_value=0, value=3, step=1)
avg_session = st.number_input("Avg Session Duration (Minutes)", min_value=0.0, value=30.0, step=1.0)
player_level = st.number_input("Player Level", min_value=0, value=5, step=1)
achievements = st.number_input("Achievements Unlocked", min_value=0, value=10, step=1)

# Predict button
if st.button("Predict Churn Risk"):
    prob, recommendation = predict_churn(
        age, gender, location, game_genre, playtime, purchases,
        game_difficulty, sessions, avg_session, player_level, achievements
    )
    st.success(f"Churn Risk Probability: {prob}")
    st.info(f"Retention Recommendation: {recommendation}")
