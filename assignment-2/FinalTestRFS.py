"""
Author: feilong.zhou, ran.he, xiaoxing.lin, tianbo.qin, deng.wei
Date: 2024-11-14
Description: Assignment-2 for regression method using Linear Regression to predict RelapseFreeSurvival (outcome)
"""

import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.pipeline import make_pipeline


# ------------------------------
# Data Preprocessing
# ------------------------------
training_file_path = 'traning_data/TrainDataset2024.xls'
data = pd.read_excel(training_file_path)

# Remove duplicates and unnecessary columns
data.drop_duplicates(inplace=True)
data.drop(columns=["ID", "pCR (outcome)"], inplace=True)

# Handle missing values by replacing with the most frequent value
data.replace(999, np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


# ------------------------------
# Feature Selection
# ------------------------------
non_numeric_feature = data.select_dtypes(include=['object'])
numeric_feature = data.select_dtypes(include=['number'], exclude=['object'])
selected_feature = numeric_feature.columns.tolist()
selected_feature.pop(0)
# print(selected_feature)  # Remove for production deployment
joblib.dump(selected_feature, "selected_feature.model")


# Select numerical features and scale the data
selected_feature = data_filled.select_dtypes(include=['number']).columns.tolist()
selected_feature.remove('RelapseFreeSurvival (outcome)')
scaler = MinMaxScaler()
data_filled[selected_feature] = scaler.fit_transform(data_filled[selected_feature])
joblib.dump(scaler, "scaler.model")

# ------------------------------
# ML Method Development
# ------------------------------
# Separate features and target variable
X = data_filled[selected_feature]
y = data_filled['RelapseFreeSurvival (outcome)'].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "linear_regression.model")

# ------------------------------
# Method Evaluation
# ------------------------------
# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# ------------------------------
# Polynomial Linear Regression
# ------------------------------
model = make_pipeline(PolynomialFeatures(3), LinearRegression())
model.fit(X_train, y_train)
joblib.dump(model, "lr.model")

# Evaluate the polynomial regression model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f"R²: {r2}")
print(f'Mean Absolute Error: {mae}')


# ------------------------------
# Test Set Prediction
# ------------------------------
scaler = joblib.load("scaler.model")
model = joblib.load("lr.model")
columns_numeric = joblib.load("selected_feature.model")
print(columns_numeric)

test_file_path = 'test_data/TestDatasetExample.xls'
test_data = pd.read_excel(test_file_path)

# Retain ID column for reporting purposes
id_list = test_data["ID"].tolist()
print(id_list)

# Remove unnecessary columns and handle missing values
test_data.drop(columns=["ID"], inplace=True)
test_data.replace(999, np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
test_data_filled = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)

# Scale test data
test_data_filled[selected_feature] = scaler.transform(test_data_filled[selected_feature])

# Ensure test data includes only selected features
X_test_final = test_data_filled[selected_feature]

# Make predictions using the trained model
y_pred_test = model.predict(X_test_final)

# Save predictions
output = pd.DataFrame({'ID': id_list, 'Prediction': y_pred_test})
output_file_path = f'predict_data/rfs/Prediction_rfs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\nPrediction results saved to {output_file_path}")
print(f"Final results:\n{output}")
