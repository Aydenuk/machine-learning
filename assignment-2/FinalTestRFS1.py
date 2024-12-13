import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.decomposition import PCA
import joblib
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

import matplotlib.pyplot as plt
import pandas as pd
import warnings
import joblib
import datetime

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

df_source = pd.read_excel('traning_data/TrainDataset2024.xls', sheet_name="Sheet1")

df_source = df_source.drop_duplicates(subset=None,keep='first')
df_source.replace(999, pd.NA, inplace=True)
df_source.dropna(inplace=True)
# df_source.duplicated().value_counts()
df_source = df_source[df_source["RelapseFreeSurvival (outcome)"] != 999]
df_source = df_source.drop("ID", axis=1).drop("pCR (outcome)", axis=1)

df_categorical_cols = df_source.select_dtypes(include=['object'])
# print(df_categorical_cols.columns.tolist())
# choose number type columns
df_numeric_cols = df_source.select_dtypes(include=['number'], exclude=['object'])
columns_numric = df_numeric_cols.columns.tolist()
columns_numric.pop(0)
print(columns_numric)
joblib.dump(columns_numric, "columns_numric.model")

scaler = MinMaxScaler()
df = df_source
df[columns_numric] = scaler.fit_transform(df[columns_numric])
df.head(100)
joblib.dump(scaler,"scaler.model")

X = df.loc[:,~df.columns.isin(['RelapseFreeSurvival (outcome)'])]
y = df.loc[:,df.columns.isin(['RelapseFreeSurvival (outcome)'])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create model
model = LinearRegression()
# train
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evalute
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae_new = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae_new}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

model = make_pipeline(PolynomialFeatures(3), LinearRegression())

# train
model.fit(X_train, y_train)


# predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# R²
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2}")
# MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


scaler = joblib.load("scaler.model")
model = joblib.load("lr.model")
columns_numric = joblib.load("columns_numric.model")

df_new = pd.read_excel('test_data/TestDatasetExample.xls', sheet_name="Sheet1")
id = df_new["ID"].tolist()
print(id)
# 预处理新数据集
df_new.replace(999, 0, inplace=True)
df_new = df_new.drop("ID", axis=1)


df_new[columns_numric] = scaler.transform(df_new[columns_numric])
# 选择特征列
X = df_new
# 使用模型进行预测
y_pred = model.predict(X)

# save predict result to csv
output = pd.DataFrame({'ID': id, 'Prediction': y_pred})
output_file_path = f'predict_data/rfs/Prediction_rfs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
output.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"\n预测结果已保存至 {output_file_path}")
print(f"最终的结果是:\n{y_pred}")
