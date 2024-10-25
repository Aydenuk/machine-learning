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

# --- 数据加载和预处理 ---

# 加载 iris 数据集
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
X_iris = iris_df.iloc[:, :-1].values  # 特征
y_iris = iris_df.iloc[:, -1].values  # 目标变量

# 加载 wine 数据集
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
X_wine = wine_df.iloc[:, :-1].values  # 特征
y_wine = wine_df.iloc[:, -1].values  # 目标变量


# --- 模型训练和评估 ---

def evaluate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=kf)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    return accuracy, precision, recall, f1


# 定义模型
models = {
    "逻辑回归": LogisticRegression(max_iter=300),
    "支持向量机": SVC(),
    "决策树": DecisionTreeClassifier(),
    "多层感知器": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
}

# 数据缩放
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# 评估Iris数据集的模型
results_iris = {}
for name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_iris_scaled, y_iris)
    results_iris[name] = (accuracy, precision, recall, f1)

# 打印Iris数据集的结果
for name, metrics in results_iris.items():
    print(
        f"{name}: 准确率 = {metrics[0]:.4f}, 精确率 = {metrics[1]:.4f}, 召回率 = {metrics[2]:.4f}, F1-score = {metrics[3]:.4f}")


# --- 可视化（示例）---
algorithms = list(results_iris.keys())
mean_accuracy = [metrics[0] for metrics in results_iris.values()]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, mean_accuracy)
plt.xlabel("算法")
plt.ylabel("准确率")
plt.title("Iris数据集算法比较")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者其他支持UTF-8的字体
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 您可以添加Wine数据集的类似可视化。





