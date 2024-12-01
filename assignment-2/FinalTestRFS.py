"""
Author: Ayden et al.
Date: 2024-11-14
Description: assignment-2 for regression method
"""
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# 主题： 利用线性回归模型来预测relapsefreesurvival_prediction值

# ------------------------------
# Data Preprocessing (10%)
# ------------------------------
print("Data processing模块从这里开始")  # 正式上线时候删除
training_file_path = 'traning_data/TrainDataset.xls'
data = pd.read_excel(training_file_path)

# data_replace = data.replace(999, np.nan, inplace=True)
# 填充缺失值
imputer = SimpleImputer(strategy='mean')
data_numeric = data.select_dtypes(include=[np.number])
data_filled_numeric = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

# 这里我将表中的value分为了数值和非数值进行处理
data_non_numeric = data.select_dtypes(exclude=[np.number])
imputer_non_numeric = SimpleImputer(strategy='most_frequent')
data_filled_non_numeric = pd.DataFrame(imputer_non_numeric.fit_transform(data_non_numeric), columns=data_non_numeric.columns)
# 合并数值和非数值数据
data_filled = pd.concat([data_filled_numeric, data_filled_non_numeric], axis=1)

# ------------------------------
# Feature Selection (25%)
# ------------------------------
print("Feature selection模块从这里开始")  # 正式上线时候删除
# 将999替换为列中的frequent value
columns_to_replace = [
    'PgR', 'HER2',
    'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType',
    'LNStatus', 'TumourStage', 'Gene'
]
# 填充缺失值，这里我将nan全部使用最频繁值来填充
imputer = SimpleImputer(missing_values=999, strategy='most_frequent')
data[columns_to_replace] = pd.DataFrame(imputer.fit_transform(data[columns_to_replace]), columns=columns_to_replace)

# 获取除了ID和两个预测值外的其他feature作为预测feature
excluded_feature = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)']
selected_feature = [col for col in data.columns if col not in excluded_feature]


# 分离特征和目标变量，这里的ID不应该包含在内
X = data_filled[selected_feature]
y_pcr = data_filled['pCR (outcome)']
y_rfs = data_filled['RelapseFreeSurvival (outcome)']
print(f"当前用于预测的列:{selected_feature}")  # 上线的时候删除

# ------------------------------
# ML Method Development (25%)
# ------------------------------
X_train_pcr, X_test_pcr, y_train_pcr, y_test_pCR = train_test_split(X, y_pcr, test_size=0.2, random_state=42)
model_pcr = LinearRegression()
model_pcr.fit(X_train_pcr, y_train_pcr)

# 试试标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 替换原始特征
X = pd.DataFrame(X_scaled, columns=X.columns)
print(f"替换后的特征矩阵已应用。")


# 训练 RelapseFreeSurvival (outcome) 的模型
print("Training model for RelapseFreeSurvival (outcome)")
X_train_rfs, X_test_rfs, y_train_rfs, y_test_rfs = train_test_split(X, y_rfs, test_size=0.2, random_state=42)
model_rfs = LinearRegression()
model_rfs.fit(X_train_rfs, y_train_rfs)

# ------------------------------
# Method Evaluation (10%)
# ------------------------------
y_pred_rfs = model_rfs.predict(X_test_rfs)
# 均绝对误差
mae_rfs = mean_absolute_error(y_test_rfs, y_pred_rfs)
# 均方误差
mse_rfs = mean_squared_error(y_test_rfs, y_pred_rfs)
# 均方根误差
rmse_rfs = np.sqrt(mse_rfs)
# 决定系数
r2_rfs = r2_score(y_test_rfs, y_pred_rfs)


print(f"\nModel Evaluation for rfs(outcome):")
print(f"Mean Squared Error (MSE) for RelapseFreeSurvival: {mse_rfs:.4f}")
print(f"Mean Absolute Error (MAE) for RelapseFreeSurvival: {mae_rfs:.4f}")
print(f"Root Mean Squared Error (RMSE) for RelapseFreeSurvival: {rmse_rfs:.4f}")
print(f"R² Score for RelapseFreeSurvival: {r2_rfs:.4f}")

# ------------------------------
# Test Set Prediction (30%)
# ------------------------------
test_file_path = 'test_data/TestDatasetExample.xls'
test_data = pd.read_excel(test_file_path)

# 确保测试集特征与训练集一致
test_data_aligned = test_data[X.columns]

# 填充缺失值
for column in X.columns:
    if test_data_aligned[column].isnull().any():
        mean_value = X_train_pcr[column].mean()  # 使用训练集均值填充
        test_data_aligned[column].fillna(mean_value, inplace=True)

# 使用训练的模型进行预测
test_predictions_pcr = model_pcr.predict(test_data_aligned)
test_predictions_rfs = model_rfs.predict(test_data_aligned)


# 初始化空列表，用于存储分行后的结果
results = []

output = pd.DataFrame({
    'pcr_prediction': test_predictions_pcr,
    'relapsefreesurvival_prediction': test_predictions_rfs
})

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# 确保每个预测值都单独填充在对应列中
output_file_path = f'predict_data/Prediction_rfs_{timestamp}.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\n预测结果已保存至 {output_file_path}")






