import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/Users/Murali krishna M/crop_yield_prediction/dataset/soil_data.csv")

# Encode the Crop column (Convert categorical values to numerical)
label_encoder = LabelEncoder()
df["Crop"] = label_encoder.fit_transform(df["Crop"])  # Encode crop names to numbers

# Split dataset into features (X) and target (y)
X = df.drop("Crop", axis=1)  # Features
y = df["Crop"]  # Target variable (Crop type)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(rf_model, "models/crop_model.pkl")
print(f"Training features: {X.columns.tolist()}")  # Add this line before training
# Save the label encoder to reuse in inference
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("✅ LabelEncoder saved successfully!")



