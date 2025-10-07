# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        input_features = [float(x) for x in request.form.values()]
        final_features = [np.array(input_features)]
        prediction = model.predict(final_features)
        result = prediction[0]
        return render_template('index.html', prediction_text=f'Predicted Species: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
