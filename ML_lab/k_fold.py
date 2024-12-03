from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ML_lab.lab import all_df
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = all_df.drop('Diagnosis', axis=1)
Xs = scaler.fit_transform(X)

Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=1, stratify=y)
SVM_clf = SVC()
y_pred = SVM_clf.fit(Xs_train, y_train).predict(Xs_test)
cm = confusion_matrix(y_test, y_pred)
