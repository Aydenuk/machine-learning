"""
Author: Ayden et al.
Date: 2024-10-29
Description: The group work of Wine
"""

import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")


wine_quality = fetch_ucirepo(id=186)
# 提取特征和目标
X = wine_quality.data.features
y = wine_quality.data.targets.values


def classify_quality(quality):
    if quality <= 4:
        return 'bad wine'
    if quality > 6:
        return 'good wine'
    else:
        return 'normal wine'


y_class = [classify_quality(quality) for quality in y]

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)
# model = SVC(kernel='linear', C=1.0, max_iter=1000)
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, Y_pred)
precision = precision_score(y_test, Y_pred, average='weighted')
recall = recall_score(y_test, Y_pred, average='weighted')
f1 = f1_score(y_test, Y_pred, average='weighted')

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
plt.title('Performance indicators of decision tree')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

