"""
Author: Ayden et al.
Date: 2024-10-25
Description: The group work of Wine
"""


# 在Jupyter Lab上可能提示你没有下载ucimlrepo， 这个时候用：!pip install ucimlrepo
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets.values


def classify_quality(quality):
    if quality <= 4:
        return 'bad wine'
    if quality > 6:
        return 'good wine'
    else:
        return 'normal wine'


# y_class = y.apply(classify_quality) panda的方法返回的会是一个pd.Series对象，不是我们想要的
y_class = [classify_quality(quality) for quality in y]

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)
# 设置模型优化算法的最大迭代次数, 过少可能会导致模型提前停止，导致未收敛
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

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

