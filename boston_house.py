import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Step 1: Load Dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Step 2: Data Exploration
print(data.head())
print(data.info())
print(data.describe())

# Step 3: Data Visualization (Correlation Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

# Step 4: Data Preprocessing
X = data.drop('PRICE', axis=1)
y = data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Building - Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 6: Model Evaluation - Linear Regression
y_pred_lr = lr.predict(X_test)
print("Linear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Step 7: Model Building - Ridge Regression
ridge = Ridge(alpha=1.0)  # Regularization strength
ridge.fit(X_train, y_train)

# Step 8: Model Evaluation - Ridge Regression
y_pred_ridge = ridge.predict(X_test)
print("\nRidge Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R2 Score:", r2_score(y_test, y_pred_ridge))

# Step 9: Visualization of Results (Horizontal Layout)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))  # 1 row, 2 columns

# First subplot: Correlation Heatmap
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=axes[0])
axes[0].set_title("Correlation Heatmap")

# Second subplot: Actual vs Predicted Prices
axes[1].scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.6, color='blue')
axes[1].scatter(y_test, y_pred_ridge, label="Ridge Regression", alpha=0.6, color='red')
axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label='Perfect Fit')
axes[1].set_xlabel("Actual Prices")
axes[1].set_ylabel("Predicted Prices")
axes[1].set_title("Actual vs Predicted Prices")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()  # Adjusts the spacing between subplots
plt.show()

