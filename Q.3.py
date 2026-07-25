import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ford.csv")

X = df.drop("price", axis=1)

categorical_columns = X.select_dtypes(include="object").columns
X = pd.get_dummies(X, columns=categorical_columns)

scaler = StandardScaler()

num_cols = ["year", "mileage", "tax", "mpg", "engineSize"]

X[num_cols] = scaler.fit_transform(X[num_cols])

print(X.head())