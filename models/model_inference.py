import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("models/crop_model.pkl")  # Ensure correct filename

# Define the crop mapping (update with actual crop names)
crop_mapping = {
    0: "Rice", 1: "Maize", 2: "Chickpea", 3: "Kidney Beans",
    4: "Pigeon Peas", 5: "Moth Beans", 6: "Mung Bean", 7: "Black Gram",
    8: "Lentil", 9: "Pomegranate", 10: "Banana", 11: "Mango",
    12: "Grapes", 13: "Watermelon", 14: "Muskmelon", 15: "Apple",
    16: "Orange", 17: "Papaya", 18: "Coconut", 19: "Cotton",
    20: "Jute", 21: "Coffee"
}

# Define feature names (same as used in training)
feature_names = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH_Value", "Rainfall"]

# Function to make a prediction
def predict_crop(features):
    """
    Predict the crop based on input features.
    :param features: List of [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall]
    :return: Predicted crop name
    """
    # Convert input list to a pandas DataFrame with feature names
    input_df = pd.DataFrame([features], columns=feature_names)  
    
    # Make prediction
    predicted_class = model.predict(input_df)[0]  
    return crop_mapping.get(predicted_class, "Unknown Crop")

# Example usage
if __name__ == "__main__":
    # Test input (replace with actual values)
    sample_input = [90, 42, 43, 20.0, 80.0, 6.5, 200.0]  # Adjust values as per dataset
    predicted_crop = predict_crop(sample_input)
    
    print(f"🌾 Recommended Crop for Given Conditions: {predicted_crop}")
    print("✅ This crop is best suited based on the provided soil and climate conditions.")
