# Step 1: Environment Setup
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow

# Step 2: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 3: Data Loading & Inspection
df = pd.read_csv('TASK-ML-INTERN.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Identify feature columns (0-447) and target
feature_columns = [str(i) for i in range(448)]
target_column = 'vomitoxin_ppb'

X = df[feature_columns]
y = df[target_column]

# Basic data check
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {X.isnull().sum().sum()}")

# Step 4: Data Preprocessing
# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 5: Spectral Visualization
plt.figure(figsize=(12, 6))
plt.plot(X.columns.astype(int), X.mean(axis=0), color='blue')
plt.title('Average Spectral Reflectance Curve')
plt.xlabel('Wavelength Band')
plt.ylabel('Reflectance (Normalized)')
plt.grid(True)
plt.show()

# Step 6: Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# PCA Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(label='DON Concentration (ppb)')
plt.title('PCA Projection of Hyperspectral Data')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.show()

print(f"Reduced from {X.shape[1]} to {pca.n_components_} components")

# Step 7: Neural Network Model
def build_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Train model
model = build_model(X_train.shape[1])
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Step 8: Model Evaluation
y_pred = model.predict(X_test).flatten()

metrics = {
    'MAE': mean_absolute_error(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'R²': r2_score(y_test, y_pred)
}

print("\nModel Performance:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual DON Concentration')
plt.ylabel('Predicted DON Concentration')
plt.title('Actual vs Predicted Values')
plt.show()

# Step 9: XGBoost Baseline
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_metrics = {
    'MAE': mean_absolute_error(y_test, xgb_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
    'R²': r2_score(y_test, xgb_pred)
}

print("\nXGBoost Performance:")
for k, v in xgb_metrics.items():
    print(f"{k}: {v:.4f}")

# Step 10: Feature Importance (XGBoost)
plt.figure(figsize=(10, 6))
sorted_idx = xgb_model.feature_importances_.argsort()[::-1][:20]
plt.barh(np.array(feature_columns)[sorted_idx][:20], 
         xgb_model.feature_importances_[sorted_idx][:20])
plt.xlabel('Importance Score')
plt.title('Top 20 Important Spectral Bands')
plt.gca().invert_yaxis()
plt.show()
