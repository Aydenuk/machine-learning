import pandas as pd
import joblib

scaler = joblib.load("scaler.model")
model = joblib.load("xgboost.model")
column_choose = joblib.load("columns_choose.model")

# load test dataset
df_new = pd.read_excel('../data/TestDatasetExample.xls',sheet_name="Sheet1")
id = df_new["ID"].tolist()
print(id)
# clean test dataset
df_new.replace(999, 0, inplace=True)
# df_new.dropna(inplace=True)
df_new = df_new.drop("ID", axis=1)
df_categorical_cols = df_new.select_dtypes(include=['object'])
df_numeric_cols = df_new.select_dtypes(include=['number'], exclude=['object'])
columns_numric = df_numeric_cols.columns.tolist()

print(columns_numric)

# scaler transform
df_new[columns_numric] = scaler.transform(df_new[columns_numric])

# choose cloumns
X = df_new[column_choose]
# print(X_new)
# predict
y_pred = model.predict(X)


# save predict result to csv
pd.DataFrame({'ID': id, 'Prediction': y_pred}).to_csv("pcr_predict.csv",index=False)
print(y_pred)


