"""
Author: Ayden et al.
Date: 2024-11-14
Description: assignment-2 for regression method
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Data processing模块
print("Data processing模块从这里开始")  # 正式上线时候删除
file_path = 'traning_data/TrainDataset.xls'
data = pd.read_excel(file_path)

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

# Feature selection模块
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


# ML Method Development模块
X_train, X_test, y_train, y_test = train_test_split(data_selected.drop(columns=['Gene']), data_selected['Gene'], test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(accuracy)
print(confusion_mat)
print(classification_rep)























