import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dataset
DATA_PATH = "data/extended_player_data.csv"
data = pd.read_csv(DATA_PATH)

# Separating feature and target variables
X = data[["Finishing", "Pass", "Technical", "Speed", "Strength", "Endurance"]]
y = data["Transfer Value (mil €)"]

# Training the Random Forest model (full data, for feature importances)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Extracting feature importances
feature_importances = rf_model.feature_importances_
features = X.columns

# Visualizing feature importances
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Features Importance")
plt.ylabel("Features")
plt.title("Feature Importance Analysis using Random Forest")
plt.show()

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Retraining the Random Forest model
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Mean Absolute Error calculation
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Transfer Value (€)")
plt.ylabel("Predicted Transfer Value (€)")
plt.title("Comparison of Actual vs Predicted Transfer Values")
plt.show()
