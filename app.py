from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/crop_model.pkl")

# Define the crop mapping
crop_mapping = {
    0: "Rice", 1: "Maize", 2: "Chickpea", 3: "Kidney Beans",
    4: "Pigeon Peas", 5: "Moth Beans", 6: "Mung Bean", 7: "Black Gram",
    8: "Lentil", 9: "Pomegranate", 10: "Banana", 11: "Mango",
    12: "Grapes", 13: "Watermelon", 14: "Muskmelon", 15: "Apple",
    16: "Orange", 17: "Papaya", 18: "Coconut", 19: "Cotton",
    20: "Jute", 21: "Coffee"
}

# Prediction function
def predict_crop(features):
    input_array = np.array(features).reshape(1, -1)
    predicted_class = model.predict(input_array)[0]
    return crop_mapping.get(predicted_class, "Unknown Crop")

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to get predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        features = [
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        # Predict the crop
        predicted_crop = predict_crop(features)

        # Render the result on the same page
        return render_template('index.html', prediction=predicted_crop)

    except Exception as e:
        return render_template('index.html', prediction="⚠️ Error: Invalid Input! Please check your values.")

if __name__ == "__main__":
    app.run(debug=True)
