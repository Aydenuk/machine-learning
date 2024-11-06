#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import mean_squared_error

# --- 数据加载和预处理 ---

# 加载 iris 数据集
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
X_iris = iris_df.iloc[:, :-1].values  # 特征
y_iris = iris_df.iloc[:, -1].values  # 目标变量


# --- 模型训练和评估 ---

def evaluate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    MST = []
    for train, test in kf.split(X, y): 
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        MST.append(mean_squared_error(y_pred,y[test]))     # 计算单个交叉的MST
    return np.mean(MST)                                    # 返回每次交叉的MST均值


# 定义模型
models = {
    "LR": LogisticRegression(max_iter=300),
    "SVM": SVC(),
    "DT": DecisionTreeClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
}

# 数据缩放
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# 评估Iris数据集的模型
results_iris = {}
for name, model in models.items():
    MST = evaluate_model(model, X_iris_scaled, y_iris)
    results_iris[name] = MST

# 打印Iris数据集的结果
for name, metrics in results_iris.items():
    print(f"{name}: Mean Squared Error = {metrics:.4f}")

algorithms = list(results_iris.keys())
MSTs = [MST for MST in results_iris.values()]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, MSTs)
plt.xlabel("Algorithm")
plt.ylabel("Mean Squared Error")
plt.title("Iris comparison")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()






