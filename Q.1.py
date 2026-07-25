import pandas as pd

# Load dataset

df = pd.read_csv(r"C:\Users\sharv\OneDrive\Desktop\assignment 20\ford.csv")

# Independent Features
X = df.drop("price", axis=1)

# Dependent Feature
y = df["price"]

# Print shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)