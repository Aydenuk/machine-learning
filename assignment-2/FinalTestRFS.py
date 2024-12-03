"""
Author: Ayden et al.
Date: 2024-11-14
Description: assignment-2 for regression method using Linear Regression to predict RelapseFreeSurvival (outcome)
"""

# ------------------------------
# Imports (Importing Libraries)
# ------------------------------
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.pipeline import make_pipeline
import warnings
import matplotlib.pyplot as plt


# ------------------------------
# Data Preprocessing (10%)
# ------------------------------
print("Data processing模块从这里开始")  # 正式上线时候删除
training_file_path = 'traning_data/TrainDataset2024.xls'
data = pd.read_excel(training_file_path)

# 删除重复行并删除不需要的列
data.drop_duplicates(inplace=True)
data.drop(columns=["ID", "pCR (outcome)"], inplace=True)

# 填充缺失值, 使用最频繁值
data.replace(999, np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# choose object type columns
non_numeric_feature = data.select_dtypes(include=['object'])
numeric_feature = data.select_dtypes(include=['number'], exclude=['object'])
selected_feature = numeric_feature.columns.tolist()
selected_feature.pop(0)
print(selected_feature)
joblib.dump(selected_feature, "selected_feature.model")


# 选择数值型特征并进行标准化
selected_feature = data_filled.select_dtypes(include=['number']).columns.tolist()
selected_feature.remove('RelapseFreeSurvival (outcome)')
scaler = MinMaxScaler()
data_filled[selected_feature] = scaler.fit_transform(data_filled[selected_feature])
joblib.dump(scaler, "scaler.model")

# ------------------------------
# Data Preparation (数据准备)
# ------------------------------
# 分离特征和目标变量，确保目标变量不在特征集中
X = data_filled[selected_feature]
y = data_filled['RelapseFreeSurvival (outcome)'].values.ravel()  # 保证 y 为 1D 数组

# ------------------------------
# Train Linear Regression Model (线性回归模型训练)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基础线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "linear_regression.model")

# 评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# ------------------------------
# Polynomial Linear Regression (多项式线性回归)
# ------------------------------
model = make_pipeline(PolynomialFeatures(3), LinearRegression())
model.fit(X_train, y_train)
joblib.dump(model, "lr.model")

# 评估多项式模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f"R²: {r2}")
print(f'Mean Absolute Error: {mae}')


# ------------------------------
# Test Set Prediction (测试集预测)
# ------------------------------
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

scaler = joblib.load("scaler.model")
model = joblib.load("lr.model")
columns_numric = joblib.load("selected_feature.model")
print(columns_numric)

test_file_path = 'test_data/TestDatasetExample.xls'
test_data = pd.read_excel(test_file_path)

# 给 ID 留个种，保证后面报告的时候能体现
id_list = test_data["ID"].tolist()
print(id_list)
# 删除不需要的列并处理缺失值
id_list = test_data["ID"].tolist()
test_data.drop(columns=["ID"], inplace=True)
test_data.replace(999, np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
test_data_filled = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)
test_data = test_data_filled.copy()


# 使用缩放器进行特征缩放
test_data_filled[selected_feature] = scaler.transform(test_data_filled[selected_feature])
# **确保测试数据只包含特征列**
X_test_final = test_data_filled[selected_feature]
# 使用训练好的线性回归模型进行预测
y_pred_test = model.predict(X_test_final)

# 保存预测结果
output = pd.DataFrame({'ID': id_list, 'Prediction': y_pred_test})
output_file_path = f'predict_data/Prediction_rfs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\n预测结果已保存至 {output_file_path}")
print(f"最终的结果是:\n{output}")
