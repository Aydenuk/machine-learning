"""
Author: Ayden et al.
Date: 2024-11-14
Description: assignment-2 for regression method
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ------------------------------
# Data Preprocessing (10%)
# ------------------------------
print("Data processing模块从这里开始")  # 正式上线时候删除
training_file_path = 'traning_data/TrainDataset.xls'
data = pd.read_excel(training_file_path)

imputer = SimpleImputer(strategy='mean')
data_numeric = data.select_dtypes(include=[np.number])
data_filled_numeric = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)
data_non_numeric = data.select_dtypes(exclude=[np.number])
imputer_non_numeric = SimpleImputer(strategy='most_frequent')
data_filled_non_numeric = pd.DataFrame(imputer_non_numeric.fit_transform(data_non_numeric), columns=data_non_numeric.columns)
data_filled = pd.concat([data_filled_numeric, data_filled_non_numeric], axis=1)


data_with_999 = data_numeric.copy()
data_with_999[data_with_999 == 999] = np.nan
# 将NaN也就是999的值进行忽略
z_scores = stats.zscore(data_with_999, nan_policy='omit')
abs_z_scores = abs(z_scores)

# 对于Z-score大于3的部分进行处理， 同时不影响999的值
threshold = 3
data_clean = data_numeric[(abs_z_scores < threshold).all(axis=1)]

scaler = StandardScaler()
data_normalization = pd.DataFrame(scaler.fit_transform(data_clean), columns=data_clean.columns)
print(f"标准化后的数据:\n", data_normalization.head())  # 测试代码，正式上线时候删除

# ------------------------------
# Feature Selection (25%)
# ------------------------------
print("Feature selection模块从这里开始")  # 正式上线时候删除
important_features = ['ER', 'HER2', 'Gene']
data_important = data_numeric[important_features]

# 这里我先删除重要特征，剩下的其他为保留特征
X = data_numeric.drop(columns=important_features)
y = data['Gene']

selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
X_selected = pd.DataFrame(X_new, columns=selected_features)
# 特征合并
data_selected = pd.concat([data_important, X_selected], axis=1)
print("Selected Features:\n", selected_features)
print("Preprocessed Data with Selected Features:\n", data_selected.head())

# ------------------------------
# ML Method Development (25%)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(data_selected.drop(columns=['Gene']), data_selected['Gene'], test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ------------------------------
# Method Evaluation (10%)
# ------------------------------

# 这里是记载的训练集
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# 这里是加载测试集
test_file_path = 'test_data/TestDatasetExample.xls'
test_data = pd.read_excel(test_file_path)

train_columns = set(selected_features)
test_columns = set(test_data.columns)
# 找到两者中共有的列
common_columns = list(train_columns.intersection(test_columns))
test_data_aligned = test_data[common_columns]

# 手动填充缺失值，使用训练集的均值填充
for column in common_columns:
    if test_data_aligned[column].isnull().any():
        # 计算训练数据中该特征的均值
        mean_value = X_train[column].mean()  # 使用训练集的X_train来计算均值
        test_data_aligned[column].fillna(mean_value, inplace=True)  # 填充缺失值999

# 手动标准化测试数据，使用训练集的均值和标准差
mean_values = X_train[common_columns].mean()
std_values = X_train[common_columns].std()
test_data_normalized = (test_data_aligned - mean_values) / std_values

# Ayden: 如果测试集缺少某些特征，则填充这些特征列（很重要）
train_columns = X_train.columns
missing_columns = set(train_columns) - set(test_data_normalized.columns)
for missing_column in missing_columns:
    test_data_normalized[missing_column] = 0
# Ayden: 确保测试集和训练集的特征顺序一致
test_data_normalized = test_data_normalized[train_columns]


# 使用标准化后的测试数据进行预测
test_X = test_data_normalized
test_predictions = model.predict(test_X)
print(f"测试集预测结果: {test_predictions}")


if 'Gene' in test_data.columns:
    test_y = test_data['Gene']
    test_accuracy = accuracy_score(test_y, test_predictions)
    test_precision = precision_score(test_y, test_predictions, average='weighted')
    test_recall = recall_score(test_y, test_predictions, average='weighted')
    test_f1 = f1_score(test_y, test_predictions, average='weighted')

    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Test Set Precision: {test_precision:.4f}")
    print(f"Test Set Recall: {test_recall:.4f}")
    print(f"Test Set F1-Score: {test_f1:.4f}")
else:
    print("测试集中没有 'Gene' 列标签.")



















