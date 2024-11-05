from ucimlrepo import fetch_ucirepo

wine_quality_data = fetch_ucirepo(id=186)
X = wine_quality_data.data.features
y = wine_quality_data.data.targets.values
print(X)
print(y)