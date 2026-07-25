import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# R² Score
score = r2_score(y_test, y_pred)

print("R² Score:", score)

print("\nInterpretation:")
print("The R² score shows how well the Linear Regression model predicts car prices.")
print("A value closer to 1 indicates better model performance.")