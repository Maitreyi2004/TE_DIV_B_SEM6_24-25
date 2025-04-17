# File name: api.py

# Import necessary libraries
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the model (ensure the path to the model file is correct)
try:
    with open('model_xgb.pkl', 'rb') as f:  # Adjust path if needed
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file not found. Please ensure 'model_xgb.pkl' is present.")

# Define a route for API requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming request data
        data = request.json
        df = pd.DataFrame(data)  # Assuming data is a JSON object structured as a table

        # Perform the prediction
        predictions = model.predict(df)

        # Return the prediction results as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
