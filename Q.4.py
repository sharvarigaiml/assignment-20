import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("ford.csv")

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# One-Hot Encoding
cat_cols = X.select_dtypes(include="object").columns
X = pd.get_dummies(X, columns=cat_cols)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape :", y_test.shape)