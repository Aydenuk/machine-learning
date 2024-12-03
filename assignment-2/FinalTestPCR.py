"""
Author: Ayden et al.
Date: 2024-11-28
Description: Classification model to predict pCR (outcome)
"""
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy import stats

# ------------------------------
# Data Preprocessing (10%)
# ------------------------------
print("Data processing模块从这里开始")  # 正式上线时候删除
training_file_path = 'traning_data/TrainDataset.xls'
data = pd.read_excel(training_file_path)

# 填充缺失值：统一对所有列进行缺失值填充
data.replace(999, np.nan, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 输出处理后的数据
process_data = f'process_data/Processed_TrainDataset_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
data_filled.to_excel(process_data, index=False, engine='openpyxl')

# ------------------------------
# Outlier Handling (处理离群值)
# ------------------------------
# 排除 ID 和目标列
excluded_columns = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)']
columns_for_outlier = [col for col in data_filled.columns if col not in excluded_columns]

# 使用 IQR 方法处理离群值
Q1 = data_filled[columns_for_outlier].quantile(0.25)
Q3 = data_filled[columns_for_outlier].quantile(0.75)
IQR = Q3 - Q1

# 标记离群值（IQR 方法）
outliers_iqr = (data_filled[columns_for_outlier] < (Q1 - 1.5 * IQR)) | (data_filled[columns_for_outlier] > (Q3 + 1.5 * IQR))

# 用中位数替换 IQR 方法检测到的离群值
for col in columns_for_outlier:
    if outliers_iqr[col].any():
        median_value = data_filled[col].median()
        data_filled.loc[outliers_iqr[col], col] = median_value

# # 使用 Z-score 方法进一步处理离群值
# for col in columns_for_outlier:
#     # 检查列是否有足够的数据点来计算 z-scores
#     if data_filled[col].nunique() > 1:  # 至少有两个唯一值才计算 Z-score
#         try:
#             z_scores = stats.zscore(data_filled[col], nan_policy='omit')
#             outliers_zscore = np.abs(z_scores) > 3
#
#             # 用中位数替换 Z-score 方法检测到的离群值
#             if outliers_zscore.any():
#                 median_value = data_filled[col].median()
#                 data_filled.loc[outliers_zscore, col] = median_value
#
#         except Exception as e:
#             print(f"Skipping Z-score calculation for column: {col} due to error: {e}")
#     else:
#         print(f"Skipping Z-score calculation for column: {col} due to insufficient unique values")


# 输出最终处理后的数据
process_data_outlier_path = f'process_data/Processed_OutliersHandled_TrainDataset_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
data_filled.to_excel(process_data_outlier_path, index=False, engine='openpyxl')
print(f"\n预测结果已保存至 {process_data_outlier_path}")

# ------------------------------
# Feature Selection (25%)
# ------------------------------
print("Feature selection模块从这里开始")  # 正式上线时候删除
# 获取特征列，排除 ID 和目标列
selected_feature = [col for col in data_filled.columns if col not in excluded_columns]
print(f"被选中的feature包含:{selected_feature}")

# 对所有特征进行标准化
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data_filled[selected_feature]), columns=selected_feature)

# 分离特征和目标变量
y_pcr = data_filled['pCR (outcome)']

# 检查目标变量
print("Unique values in pCR (outcome):", y_pcr.unique())
# 替换 999 为目标变量中的最频繁值
y_pcr = y_pcr.replace(999, np.nan)
imputer = SimpleImputer(strategy='most_frequent')
y_pcr = pd.Series(imputer.fit_transform(y_pcr.values.reshape(-1, 1)).ravel())  # 填充缺失值并转为一维
y_pcr = y_pcr.astype(int)

print("After imputing 999, unique values in pCR (outcome):", y_pcr.unique())
print(f"Shape of X: {X.shape}, Shape of y_pcr: {y_pcr.shape}")

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
test_data_aligned = test_data[selected_feature]

# 对测试集进行标准化，使用训练集的标准化参数
test_data_aligned = pd.DataFrame(scaler.transform(test_data_aligned), columns=selected_feature)  # 标准化测试集数据

# 填充测试集的缺失值
for column in X.columns:
    if test_data_aligned[column].isnull().any():
        mean_value = X_train[column].mean()  # 使用训练集均值填充
        test_data_aligned[column].fillna(mean_value, inplace=True)

# 使用模型进行预测
test_predictions = model.predict(test_data_aligned)

# 保存预测结果（包括 ID 列以方便对照）
output = pd.DataFrame({
    'ID': test_data['ID'],
    'pcr_prediction': test_predictions
})

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = f'predict_data/Prediction_pcr_{timestamp}.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\n预测结果已保存至 {output_file_path}")
