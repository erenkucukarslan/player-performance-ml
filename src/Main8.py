#Code: Education and Test Data Preparation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset
data = pd.read_csv("extended_player_data.csv")

# Separate features and target variable
X = data[["Finishing", "Pass", "Technical", "Speed", "Strength", "Endurance"]]  # Performance metrics
y = data["Transfer Value (mil €)"]  # Target variable (transfer value)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling of data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Code: Implementing Algorithms and Performance Evaluation

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Defining models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regression": SVR()
}

results = []

# Testing each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": model_name, "MAE": mae, "R²": r2})

# Displaying results in a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Visualizing feature importance using Random Forest

import matplotlib.pyplot as plt

feature_importances = models["Random Forest"].feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.show()