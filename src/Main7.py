from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Robust dataset path
ROOT = Path(__file__).resolve().parents[1]  # repo root (one level above src/)
DATA_PATH = ROOT / "data" / "extended_player_data.csv"

data = pd.read_csv(DATA_PATH)

FEATURES = ["Finishing", "Pass", "Technical", "Speed", "Strength", "Endurance"]
TARGET = "Transfer Value (mil €)"


# Basic train/test + scaling + RandomForest regression
X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dataset Size: {data.shape}")
print(f"Training Data Size: {X_train_scaled.shape}")
print(f"Test Data Size: {X_test_scaled.shape}")

model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# 1) Real vs Estimated scatter
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Real Transfer Value (mil €)")
plt.ylabel("Estimated Transfer Value (mil €)")
plt.title("Real vs. Estimated Transfer Value (Random Forest)")
plt.show()

# Sample predictions
sample_players = X_test_scaled[:5]
predicted_values = model.predict(sample_players)

print("Sample Player Predictions:")
for i, value in enumerate(predicted_values):
    print(f"Player {i+1}: Predicted Transfer Value = {value:.2f} million €")


# 2) Correlation of features with target (pie chart)
numeric_data = data.select_dtypes(include=["float64", "int64"])
corr_series = numeric_data.corr(numeric_only=True)[TARGET].drop(labels=[TARGET], errors="ignore")

features = corr_series.index.tolist()
weights = [abs(w) for w in corr_series.values]

plt.figure(figsize=(8, 6))
plt.pie(weights, labels=features, autopct="%1.1f%%", startangle=140)
plt.title("Weights of Features According to Transfer Value")
plt.show()


# 3) K-Means Clustering
X_cluster = data[FEATURES].copy()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)

clustered_data = data.copy()
clustered_data["Cluster"] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=clustered_data,
    x="Finishing",
    y="Speed",
    hue="Cluster",
    palette="viridis"
)
plt.title("Clustering Players (Finishing vs Speed)")
plt.xlabel("Finishing")
plt.ylabel("Speed")
plt.legend(title="Cluster")
plt.show()


# 4) Transfer Value Estimation with Random Forest (test set)
# (kept as a separate section for your report structure)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_test_pred2 = rf_model.predict(X_test_scaled)
mae2 = mean_absolute_error(y_test, y_test_pred2)
print(f"Test Set MAE (Random Forest): {mae2}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred2, alpha=0.7)
plt.xlabel("Real Transfer Value (mil €)")
plt.ylabel("Estimated Transfer Value (mil €)")
plt.title("Real vs Estimated Transfer Value (Random Forest)")
plt.show()


# 5) Correlation Matrix (heatmap)
corr_matrix = numeric_data.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation of Performance Metrics")
plt.show()








