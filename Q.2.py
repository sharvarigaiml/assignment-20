import pandas as pd

df = pd.read_csv("ford.csv")

X = df.drop("price", axis=1)
y = df["price"]

# Find categorical columns
cat_cols = X.select_dtypes(include="object").columns
print(cat_cols)

# One-Hot Encoding
X = pd.get_dummies(X, columns=cat_cols)

# Convert Boolean columns into Integer
X = X.astype(int)

# First 5 rows
print(X.head())