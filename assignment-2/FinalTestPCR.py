"""
Author: Ayden et al.
Date: 2024-11-28
Description: Classification model to predict pCR (outcome)
"""
# import os
import datetime
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# ------------------------------
# Data Preprocessing (10%)
# ------------------------------
print("Data processing模块从这里开始")  # 正式上线时候删除
training_file_path = 'traning_data/TrainDataset.xls'
data = pd.read_excel(training_file_path)

# 填充数值型特征缺失值
imputer = SimpleImputer(strategy='mean')
data_numeric = data.select_dtypes(include=[np.number])
data_filled_numeric = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

# 填充非数值型特征缺失值
data_non_numeric = data.select_dtypes(exclude=[np.number])
imputer_non_numeric = SimpleImputer(strategy='most_frequent')
data_filled_non_numeric = pd.DataFrame(imputer_non_numeric.fit_transform(data_non_numeric), columns=data_non_numeric.columns)

# 合并数值和非数值特征
data_filled = pd.concat([data_filled_numeric, data_filled_non_numeric], axis=1)

# ------------------------------
# Feature Selection (25%)
# ------------------------------
print("Feature selection模块从这里开始")  # 正式上线时候删除
# 填充 999 为列中最频繁值
columns_to_replace = [
    'PgR', 'HER2',
    'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType',
    'LNStatus', 'TumourStage', 'Gene'
]
imputer = SimpleImputer(missing_values=999, strategy='most_frequent')
data[columns_to_replace] = pd.DataFrame(imputer.fit_transform(data[columns_to_replace]), columns=columns_to_replace)

# 获取特征列，排除 ID 和目标列
excluded_feature = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)']
selected_feature = [col for col in data.columns if col not in excluded_feature]

# 分离特征和目标变量
X = data_filled[selected_feature]
y_pcr = data_filled['pCR (outcome)']

# 检查目标变量
print("Unique values in pCR (outcome):", y_pcr.unique())

# 清理目标变量中非 0/1 的值
valid_indices = (y_pcr == 0) | (y_pcr == 1)  # 获取有效行索引
X = X[valid_indices]  # 保留有效行
y_pcr = y_pcr[valid_indices]  # 同步清理目标变量

# 重置索引，确保数据一致
X = X.reset_index(drop=True)
y_pcr = y_pcr.reset_index(drop=True)

# ------------------------------
# ML Method Development (25%)
# ------------------------------
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_pcr, test_size=0.2, random_state=42)

# 使用逻辑回归进行分类
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Method Evaluation (10%)
# ------------------------------
# 预测分类结果
y_pred = model.predict(X_test)

# 计算分类指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"\nClassification Metrics for pCR (outcome):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ------------------------------
# Test Set Prediction (30%)
# ------------------------------
test_file_path = 'test_data/TestDatasetExample.xls'
test_data = pd.read_excel(test_file_path)

# 确保测试集特征与训练集一致
test_data_aligned = test_data[X.columns]

# 填充测试集的缺失值
for column in X.columns:
    if test_data_aligned[column].isnull().any():
        mean_value = X_train[column].mean()  # 使用训练集均值填充
        test_data_aligned[column].fillna(mean_value, inplace=True)

# 使用模型进行预测
test_predictions = model.predict(test_data_aligned)

# 保存预测结果
output = pd.DataFrame({
    'pcr_prediction': test_predictions  # pCR 预测值
})

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = f'predict_data/Prediction_pcr_{timestamp}.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\n预测结果已保存至 {output_file_path}")
