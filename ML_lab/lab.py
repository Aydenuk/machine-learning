import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import scipy
from scipy.stats import pearsonr
import sklearn

from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

all_df = pd.read_csv('WDBC.csv', index_col=False)
all_df.head()

all_df.drop('ID', axis=1, inplace=True)
all_df.head()
all_df.info()
all_df.describe()
all_df['Diagnosis'].value_counts()

# 生成条形图检
all_df['Diagnosis'] = all_df['Diagnosis'].replace('B', 0, regex=True)
all_df['Diagnosis'] = all_df['Diagnosis'].replace('M', 1, regex=True)
sns.countplot(x="Diagnosis", data=all_df)

# 生成box plot
# data_mean = all_df.iloc[:, :]
# data_mean.plot(kind='box', subplots=True, layout=(8,4), sharex=False,
# sharey=False, fontsize=12, figsize=(15,20));

data = all_df.iloc[:, 1:11]  # 表示从 all_df 数据框中提取列索引从 1 到 10 的数据（即第 1 列到第 10 列，不包含第 11 列）。.iloc 是 Pandas 提供的按位置索引选择数据的方法
# ax=ax 表示将这个图表绘制在前面创建的 ax 坐标轴上
fig, ax = plt.subplots(1, figsize=(20, 8))
sns.boxplot(data=all_df.iloc[:, 1:11], ax=ax)

data = all_df.iloc[:, 1:11]
ax = ax  # 表示将这个图表绘制在前面创建的 ax 坐标轴上
fig, ax = plt.subplots(1, figsize=(20, 8))
sns.boxplot(data=all_df.iloc[:, 1:11], ax=ax)

# # 修改为获取11-20的数据
# fig,ax=plt.subplots(1,figsize=(20,8))
# sns.boxplot(data=all_df.iloc[:, 11:20], ax=ax)


fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(15, 20))
fig.subplots_adjust(hspace=.2, wspace=.5)
axes = axes.ravel()
for i, col in enumerate(all_df.columns[1:]):
    _ = sns.boxplot(y=col, x='Diagnosis', data=all_df, ax=axes[i])

# 从all_df文件中提取数值型列
corrMatt = all_df.corr(numeric_only=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corrMatt)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 12))
plt.title('Breast Cancer Feature Correlation')
# Generate a custom diverging colormap
cmap = sns.diverging_palette(260, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrMatt, vmax=1.2, square=False, cmap=cmap, mask=mask,
            ax=ax, annot=True, fmt='.2g', linewidths=1);

cat_encoder = OneHotEncoder()

# Assign features to X
X = all_df.drop('Diagnosis', axis=1)
# Normalise the features to use zero mean normalisation
# only for the first 10 features, but try yourself to visualise more features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
fig, ax = plt.subplots(1, figsize=(20, 8))
sns.boxplot(data=Xs, ax=ax)

Xs_centered = Xs - Xs.mean(axis=0)
U, s, Vt = np.linalg.svd(Xs_centered)
c1 = Vt.T[:, 0]  # first mode of PC
c2 = Vt.T[:, 1]  # second mode of PC
W2 = Vt.T[:, :2]  # only retain the first two principle components.
X2D = Xs_centered.dot(W2)  # project the data into PCA space
PCA_df = pd.DataFrame()
PCA_df['PCA_1'] = X2D[:, 0]
PCA_df['PCA_2'] = X2D[:, 1]

feature_names = list(X.columns)
pca = PCA(n_components=10)
Xs_pca = pca.fit_transform(Xs)
PCA_df = pd.DataFrame()
PCA_df['PCA_1'] = Xs_pca[:, 0]
PCA_df['PCA_2'] = Xs_pca[:, 1]

plt.figure(figsize=(6, 6))
plt.plot(PCA_df['PCA_1'][all_df['Diagnosis'] ==
                         1], PCA_df['PCA_2'][all_df['Diagnosis'] == 1], 'ro', alpha=0.7, markeredgecolor
         ='k')
plt.plot(PCA_df['PCA_1'][all_df['Diagnosis']
                         == 0], PCA_df['PCA_2'][all_df['Diagnosis'] == 0], 'bo', alpha=0.7,
         markeredgecolor='k')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.legend(['Malignant', 'Benign']);

le = LabelEncoder()
all_df['Diagnosis'] = le.fit_transform(all_df['Diagnosis'])
all_df.head()
# assign numerical label to y
y = all_df['Diagnosis']

# 逻辑线性回归
Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=1, stratify=y)
log_reg = LogisticRegression()
log_reg.fit(Xs_train, y_train)
classifier_score = log_reg.score(Xs_test, y_test)
print('The classifier of linear regression accuracy score is {:03.2f}'.format(classifier_score))

# 高斯朴素贝叶斯
gnb_clf = GaussianNB()
gnb_clf.fit(Xs_train, y_train)
classifier_score = gnb_clf.score(Xs_test, y_test)
print('The classifier of naive byers accuracy score is {:03.2f}'.format(classifier_score))

# K邻
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(Xs_train, y_train)
classifier_score = knn_clf.score(Xs_test, y_test)
print('The classifier of K-Nearest Neighbour accuracy score is {:03.2f}'.format(classifier_score))

# lab4:  more linear model

# ***使用决策树的预测模型***
# 创建一个最大深度为2的决策树分类器。这意味着树的最大层数为2，这通常用于防止过拟合。
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(Xs_train, y_train)
# 预测测试集的结果
tree_clf.predict_proba(Xs_test)

# 使用测试集 Xs_test 和真实标签 y_test 来计算模型的准确率
classifier_score = tree_clf.score(Xs_test, y_test)
print('The classifier accuracy score of Decision Tree is {:03.2f}'.format(classifier_score))
# 可视化决策树
tree.plot_tree(tree_clf)

# ***使用随机森林的预测模型***
# 指定森林中的树为500颗，每棵树的最大叶节点是10防止过度拟合，并使用所有可用CPU来进行计算
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=10, n_jobs=-1)
rnd_clf.fit(Xs_train, y_train)
y_pred_rf = rnd_clf.predict(Xs_test)
# 评估准确率
classifier_score = rnd_clf.score(Xs_test, y_test)
print('The classifier accuracy score of Random Forest is {:03.2f}'.format(classifier_score))

importances = rnd_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rnd_clf.estimators_],
             axis=0)
feature_names = all_df.columns[1:]
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importance using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# MLP预测模型
mlp_clf = MLPClassifier(random_state=1, max_iter=300)
mlp_clf.fit(Xs_train, y_train)
mlp_clf.predict_proba(Xs_test)
classifier_score = tree_clf.score(Xs_test, y_test)
print('The classifier accuracy score of MLP is {:03.2f}'.format(classifier_score))

# lab 5 -> K-fold交叉验证
clf_cv = SVC()
scores = cross_val_score(clf_cv, Xs, y, cv=5)
print(f"Score is:{scores}")
avg = (100 * np.mean(scores), 100 * np.std(scores) / np.sqrt(scores.shape[0]))
print("Average score and standard deviation: (%.2f +- %.3f)%%" % avg)

kf = KFold()
for train, test in(kf.split(Xs,y)):
    SVM_clf = SVC()
    SVM_clf.fit(Xs[train], y[train])
    classifier_score = SVM_clf.score(Xs[test], y[test])
    print('The classifier accuracy is {:03.2f}'.format(classifier_score))




