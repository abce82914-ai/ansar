# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("IRIS.csv")

# Drop unnecessary columns if any
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Split features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model to file
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
