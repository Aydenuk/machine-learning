"""
Author: Ayden et al.
Date: 2024-10-29
Description: The group work of Wine
"""
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets.values


def classify_quality(quality):
    if quality <= 4:
        return 0  # bad wine
    if quality > 6:
        return 1  # good wine
    else:
        return 2  # normal wine


# 将目标转换成分类
y_class = np.array([classify_quality(quality) for quality in y])
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=44)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1, validation_split=0.1)
history = model.fit(X_train, y_train, epochs=10, batch_size=5, verbose=1, validation_split=0.1)

Y_pred = np.argmax(model.predict(X_test), axis=-1)

accuracy = accuracy_score(y_test, Y_pred)
precision = precision_score(y_test, Y_pred, average='weighted')
recall = recall_score(y_test, Y_pred, average='weighted')
f1 = f1_score(y_test, Y_pred, average='weighted')

print(f"The accuracy of this model is {accuracy}")
print(f"The precision of this model is {precision}")
print(f"The recall of this model is {recall}")
print(f"The f1 of this model is {f1}")

# 绘制训练过程中的损失和准确率
plt.figure(figsize=(12, 5))

matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red'])
plt.ylabel('Value')
plt.title('Performance indicators of neural network')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
