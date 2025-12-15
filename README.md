🎮 Player Churn Prediction

This project focuses on predicting player churn in online games using a Logistic Regression model. Player churn, or when a player stops engaging with a game, is a critical metric 
for game developers because retaining players is often more cost-effective than acquiring new ones. The goal of this project is to identify high-risk players based on their 
in-game behavior and provide actionable recommendations to improve retention.

The dataset used contains information about player demographics, gameplay behavior, and engagement metrics, such as age, gender, location, game genre, playtime hours, 
in-game purchases, game difficulty, session frequency, average session duration, player level, and achievements unlocked. Extensive data exploration was performed to 
understand the distributions of features, correlations with churn, and patterns in high-risk players. This helped in engineering meaningful features and preparing the 
data for modeling.

A Logistic Regression model was trained on the processed data. Categorical features were encoded using one-hot encoding, and numerical features were scaled for better 
model performance. The model predicts the probability of churn for individual players. Based on the predicted probability, players are categorized into low, medium, 
or high risk, with retention recommendations provided for each risk level.

To make the model accessible and interactive, a Streamlit web app was developed. Users can input player information through the interface and receive a predicted 
churn probability along with actionable retention strategies. This makes it easier for game developers or analysts to quickly identify at-risk players and take 
preventive measures.

Future improvements could include testing more advanced models like Random Forest or XGBoost, incorporating time-series data for sequential 
behavior analysis, or creating a more dynamic dashboard for real-time player monitoring. Overall, this project demonstrates the practical application of machine learning 
in gaming analytics and retention strategies.
