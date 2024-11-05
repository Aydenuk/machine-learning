"""
Author: Ayden et al.
Date: 2024-10-25
Description: The group work of Wine
"""


# 在Jupyter Lab上可能提示你没有下载ucimlrepo， 这个时候用：!pip install ucimlrepo
import matplotlib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import os
import warnings
import seaborn as sns 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


wine_quality = fetch_ucirepo(id=186)
#设置X，y
X = wine_quality.data.original.drop("color",axis = 1)
y = wine_quality.data.original["color"]

# Normalise the features to use zero mean normalisation 提升数据稳定性优化模型 
scaler = StandardScaler() 
Xs = scaler.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)

# 设置模型优化算法的最大迭代次数, 过少可能会导致模型提前停止，导致未收敛
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 模拟k-fold功能
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_value_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_weighted')
# 输出交叉验证结果
# 打印每个折叠的 F1 分数及其均值和标准差
# print(f"Cross-validated F1 scores: {cross_value_scores}")
# print(f"Mean F1 score: {np.mean(cross_value_scores)}")
# print(f"Standard deviation of F1 scores: {np.std(cross_value_scores)}")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


print(f"The accuracy of this model is {accuracy}")
print(f"The precision of this model is {precision}")
print(f"The recall of this model is {recall}")
print(f"The f1 of this model is {f1}")

metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}


matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red'])
plt.ylabel('Value')
plt.title('Performance indicators of linear regression')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
