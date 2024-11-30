"""
Author: Ayden et al.
Date: 2024-11-14
Description: assignment-2 for regression method
"""
import os
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

data_non_numeric = data.select_dtypes(exclude=[np.number])
imputer_non_numeric = SimpleImputer(strategy='most_frequent')
data_filled_non_numeric = pd.DataFrame(imputer_non_numeric.fit_transform(data_non_numeric),
                                       columns=data_non_numeric.columns)
# 合并数值和非数值数据
data_filled = pd.concat([data_filled_numeric, data_filled_non_numeric], axis=1)

# ------------------------------
# Feature Selection (25%)
# ------------------------------
print("Feature selection模块从这里开始")  # 正式上线时候删除
data_with_999 = data_numeric.copy()
data_with_999[data_with_999 == 999] = np.nan
# 将NaN也就是999的值进行忽略
z_scores = stats.zscore(data_with_999, nan_policy='omit')
abs_z_scores = abs(z_scores)

# 将指定列的替换999值为Nan
columns_to_replace = [
    'ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'PgR', 'HER2',
    'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType',
    'LNStatus', 'TumourStage', 'Gene'
]
data[columns_to_replace] = data[columns_to_replace].replace(999, np.nan)
# 填充缺失值，这里我将nan全部使用最频繁值来填充
imputer = SimpleImputer(strategy='most_frequent')
data[columns_to_replace] = imputer.fit_transform(data[columns_to_replace])

# 对于Z-score大于3的部分进行处理， 同时不影响999的值
threshold = 3
data_clean = data_numeric[(abs_z_scores < threshold).all(axis=1)]

scaler = StandardScaler()
data_normalization = pd.DataFrame(scaler.fit_transform(data_clean), columns=data_clean.columns)
print(f"标准化后的数据:\n", data_normalization.head())

# 分离特征和目标变量，这里的ID不应该包含在内
X = data_filled.drop(columns=['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)'])
y_pcr = data_filled['pCR (outcome)']
y_rfs = data_filled['RelapseFreeSurvival (outcome)']


# ------------------------------
# ML Method Development (25%)
# ------------------------------
# 训练pCR(outcome)的模型
X_train_pcr, X_test_pcr, y_train_pcr, y_test_pCR = train_test_split(X, y_pcr, test_size=0.2, random_state=42)
model_pcr = LinearRegression()
model_pcr.fit(X_train_pcr, y_train_pcr)

# 训练 RelapseFreeSurvival (outcome) 的模型
print("Training model for RelapseFreeSurvival (outcome)")
X_train_rfs, X_test_rfs, y_train_rfs, y_test_rfs = train_test_split(X, y_rfs, test_size=0.2, random_state=42)
model_rfs = LinearRegression()
model_rfs.fit(X_train_rfs, y_train_rfs)

# ------------------------------
# Method Evaluation (10%)
# ------------------------------
# 这里是记载的训练集
# 评估 pCR (outcome)
y_pred_pcr = model_pcr.predict(X_test_pcr)
mae_pcr = mean_absolute_error(y_test_pCR, y_pred_pcr)
mse_pcr = mean_squared_error(y_test_pCR, y_pred_pcr)
rmse_pcr = np.sqrt(mse_pcr)
r2_pCR = r2_score(y_test_pCR, y_pred_pcr)

print(f"\nModel Evaluation for pCR (outcome):")
print(f"MAE: {mae_pcr:.4f}, MSE: {mae_pcr:.4f}, RMSE: {rmse_pcr:.4f}, R²: {r2_pCR:.4f}")


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
    'pcr_prediction': test_predictions_pcr,  # pCR预测值
    'relapsefreesurvival_prediction': test_predictions_rfs  # RFS预测值
})

# 确保每个预测值都单独填充在对应列中
output_file_path = 'predict_data/Prediction_pcr.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\n预测结果已保存至 {output_file_path}")


#
# # 这里是加载测试集
# test_file_path = 'test_data/TestDatasetExample.xls'
# test_data = pd.read_excel(test_file_path)
#
# train_columns = set(selected_features)
# test_columns = set(test_data.columns)
# # 找到两者中共有的列
# common_columns = list(train_columns.intersection(test_columns))
# test_data_aligned = test_data[common_columns]
#
# # 手动填充缺失值，使用训练集的均值填充
# for column in common_columns:
#     if test_data_aligned[column].isnull().any():
#         # 计算训练数据中该特征的均值
#         mean_value = X_train[column].mean()  # 使用训练集的X_train来计算均值
#         test_data_aligned[column].fillna(mean_value, inplace=True)  # 填充缺失值999
#
# # 手动标准化测试数据，使用训练集的均值和标准差
# mean_values = X_train[common_columns].mean()
# std_values = X_train[common_columns].std()
# test_data_normalized = (test_data_aligned - mean_values) / std_values
#
# # Ayden: 如果测试集缺少某些特征，则填充这些特征列（很重要）
# train_columns = X_train.columns
# # print(X_train.columns)
# missing_columns = set(train_columns) - set(test_data_normalized.columns)
# # 对于缺失的内容我全部填写0
# for missing_column in missing_columns:
#     test_data_normalized[missing_column] = 0
# # Ayden: 确保测试集和训练集的特征顺序一致
# test_data_normalized = test_data_normalized[train_columns]
# # 使用标准化后的测试数据进行预测
# test_X = test_data_normalized
# test_predictions = model.predict(test_X)
# print(f"测试集预测结果: {test_predictions}")
#
# if 'Gene' in test_data.columns:
#     test_y = test_data['Gene']
#     test_accuracy = accuracy_score(test_y, test_predictions)
#     test_precision = precision_score(test_y, test_predictions, average='weighted')
#     test_recall = recall_score(test_y, test_predictions, average='weighted')
#     test_f1 = f1_score(test_y, test_predictions, average='weighted')
#
#     print(f"Test Set Accuracy: {test_accuracy:.4f}")
#     print(f"Test Set Precision: {test_precision:.4f}")
#     print(f"Test Set Recall: {test_recall:.4f}")
#     print(f"Test Set F1-Score: {test_f1:.4f}")
# else:
#     print("测试集中没有 'Gene' 列标签.")
