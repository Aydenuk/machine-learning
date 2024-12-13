"""
Author: feilong.zhou, ran.he, xiaoxing.lin, tianbo.qin, deng.wei
Date: 2024-11-28
Description: Classification model to predict pCR (outcome) using Random Forest and SVM
"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# Data Preprocessing
# ------------------------------
print("Data processing module starts here")  # Remove before production
training_file_path = 'traning_data/TrainDataset2024.xls'
data = pd.read_excel(training_file_path)

data.drop_duplicates(inplace=True)
data.drop(columns=["ID", "RelapseFreeSurvival (outcome)"], inplace=True)
data.replace(999, np.nan, inplace=True)

imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
data = data_filled.copy()

# ------------------------------
# Feature Selection (25%)
# ------------------------------
# Feature Scaling (Data Normalization)
# Get all numeric columns from the dataframe, excluding the target column 'pCR (outcome)'
df_numeric_cols = data.select_dtypes(include=['number'])
columns_numeric = [col for col in df_numeric_cols.columns if col != 'pCR (outcome)']

if columns_numeric:  # If numeric columns exist
    scaler = MinMaxScaler()
    data[columns_numeric] = scaler.fit_transform(data[columns_numeric])
    joblib.dump(scaler, "scaler.model")

    data_with_id = data_filled.copy()
    process_data_outlier_path = f'process_data/Processed_Normalized_TrainDataset_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    data.to_excel(process_data_outlier_path, index=False, engine='openpyxl')
    print(f"\nNormalized data has been saved to {process_data_outlier_path}")
else:
    raise ValueError("No numeric columns found for scaling.")


# Data Preparation
# Separate features and target variable, ensuring the target variable is not in the features
X = data.loc[:, ~data.columns.isin(['pCR (outcome)'])]
y = data['pCR (outcome)']


# Data Balancing
ros = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

# ------------------------------
# ML Method Development
# ------------------------------
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Use Random Forest for feature selection
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Select Top 50 features, including Gene, ER, and HER2 features
sorted_index = model_rf.feature_importances_.argsort()
columns_sort = X_train.columns[sorted_index]
base_columns = ["Gene", "ER", "HER2"]
columns_choose = columns_sort[-50:].tolist()
columns_choose = list(set(columns_choose) | set(base_columns))
joblib.dump(columns_choose, "columns_choose.model")
print(f"Selected features are {columns_choose}")

# Visualization of Feature Importance
plt.figure(figsize=(10, 20))
plt.barh(range(len(columns_sort)), model_rf.feature_importances_[sorted_index])
plt.yticks(np.arange(len(columns_sort)), columns_sort)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Distribution Chart')
plt.tight_layout()
plt.show()

# Save the Random Forest model
joblib.dump(model_rf, "rf.model")

# Final Model Training - SVM
# Use selected features for training
X_train_final = X_train[columns_choose]
X_test_final = X_test[columns_choose]

# Use SVM for classification
model_svm = SVC(kernel='linear', probability=True)
model_svm.fit(X_train_final, y_train)
joblib.dump(model_svm, 'svm.model')


# ------------------------------
# Method Evaluation
# ------------------------------
# Predict and print evaluation metrics
y_pred = model_svm.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("\nDetailed Metrics for pCR (outcome):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix Visualization
confusion_matrix_result = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

# ------------------------------
# Test Set Prediction
# ------------------------------
# Load the test dataset
test_file_path = 'test_data/TestDatasetExample.xls'
test_data = pd.read_excel(test_file_path)

# Drop unnecessary columns and handle missing values
id_list = test_data["ID"].tolist()
test_data.drop(columns=["ID"], inplace=True)
test_data.replace(999, np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
test_data_filled = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)
test_data = test_data_filled.copy()

# Select numeric columns and perform scaling
df_numeric_cols_test = test_data.select_dtypes(include=['number'])
columns_numeric_test = df_numeric_cols_test.columns.tolist()

if columns_numeric_test:  # If numeric columns exist, perform scaling
    scaler = joblib.load("scaler.model")
    test_data[columns_numeric_test] = scaler.transform(test_data[columns_numeric_test])

# Select columns chosen during feature selection for prediction
X_test_final = test_data.reindex(columns=columns_choose, fill_value=0)

# Predict using the SVM model
y_pred_test = model_svm.predict(X_test_final)

# Save prediction results
output = pd.DataFrame({'ID': id_list, 'Prediction': y_pred_test})
output_file_path = f'predict_data/pcr/Prediction_pcr_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\nPrediction results have been saved to {output_file_path}")
print(f"The final results are:\n{output}")
