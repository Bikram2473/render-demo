from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model
try:
    model = joblib.load('sentiment_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'sentiment_model.pkl' not found. Train the model first.")
    exit(1)

# Define the /analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get JSON data from the request
        data = request.json
        text = data['text']

        # Predict sentiment
        prediction = model.predict([text])[0]
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        # Return JSON response
        return jsonify({
            'text': text,
            'sentiment': sentiment_map[prediction],
            'confidence': float(np.max(model.predict_proba([text])))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run on port 5000