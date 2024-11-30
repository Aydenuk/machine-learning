"""
Author: Ayden et al.
Date: 2024-11-28
Description: assignment-2 for classification
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_excel('traning_data/TrainDataset.xls')

columns_to_replace = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'PgR', 'HER2', 'TrippleNegative',
                      'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage', 'Gene']
data[columns_to_replace] = data[columns_to_replace].replace(999, np.nan)

# 对所有列使用最频繁值填充NaN
imputer_non_numeric = SimpleImputer(strategy='most_frequent')
data_filled_non_numeric = pd.DataFrame(imputer_non_numeric.fit_transform(data), columns=data.columns)

# 将pCR (outcome)列转为二分类（如果它不是 0 和 1）
# 假设pCR (outcome)列中的目标值是0和1，其他的NaN或非0/1值都替换成最频繁值
data_filled_non_numeric['pCR (outcome)'] = data_filled_non_numeric['pCR (outcome)'].apply(lambda x: 1 if x == 0 else 0)

# 分离特征和目标变量
X = data_filled_non_numeric.drop(columns=['ID', 'pCR (outcome)'])  # ID 和 pCR (outcome) 不作为特征
y = data_filled_non_numeric['pCR (outcome)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {f1:.4f}")




# 这里是加载测试集
test_file_path = 'test_data/TestDatasetExample.xls'
test_data = pd.read_excel(test_file_path)

# 填充测试集的缺失值，替换999为NaN
test_data[columns_to_replace] = test_data[columns_to_replace].replace(999, np.nan)
test_data_filled = pd.DataFrame(imputer_non_numeric.transform(test_data), columns=columns_to_replace)

# 将测试集的pCR (outcome)列转为二分类
test_data_filled['pCR (outcome)'] = test_data_filled['pCR (outcome)'].apply(lambda x: 1 if x == 0 else 0)

# 标准化测试数据
X_test_data = test_data_filled.drop(columns=['ID', 'pCR (outcome)'])
X_test_scaled_data = scaler.transform(X_test_data)

test_predictions = model.predict(X_test_scaled_data)

print(f"测试集预测结果: {test_predictions}")

if 'pCR (outcome)' in test_data.columns:
    y_test_data = test_data['pCR (outcome)']
    test_accuracy = accuracy_score(y_test_data, test_predictions)
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
