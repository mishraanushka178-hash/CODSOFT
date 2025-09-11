import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys

# Path to dataset
dataset_path = "data/IMDB Movies India.csv"

try:
    df = pd.read_csv(dataset_path, encoding="utf-8", on_bad_lines="skip")
    print("✅ Dataset loaded successfully with UTF-8 encoding")
except UnicodeDecodeError:
    df = pd.read_csv(dataset_path, encoding="latin-1", on_bad_lines="skip")
    print("✅ Dataset loaded successfully with latin-1 encoding")
except FileNotFoundError:
    print(f"❌ Dataset not found at {dataset_path}. Please make sure the file exists.")
    sys.exit(1)

# 🔹 Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "").str.replace("-", "")

print("\n📊 Dataset Preview:")
print(df.head())

# Check if required columns exist
required_columns = ["imdb_rating", "votes"]
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print(f"❌ Required columns missing: {missing_cols}")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

# Define features (X) and target (y)
X = df[["votes"]]
y = df["imdb_rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n📈 Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))