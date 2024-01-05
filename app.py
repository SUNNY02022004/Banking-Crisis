from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('banking_crisis_model.joblib')

# Define the feature names
feature_names = [f'feature_{i}' for i in range(10)]  # Update the range to 10

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Initialize variables
        features = []
        error_message = None

        # Extract features from the form and validate input
        for i in range(10):  # Update the range to 10
            try:
                value = float(request.form[f'feature_{i}'])
                features.append(value)
            except ValueError:
                error_message = f"Error: Feature {i+1} must be a numeric value."
                return redirect(url_for('error', error_message=error_message))

        # Create a DataFrame from the input data with correct column names
        input_data = pd.DataFrame([features], columns=feature_names)
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Convert binary prediction to 'Yes' or 'No'
        prediction_result = 'Yes' if prediction == 1 else 'No'
        
        # Render the prediction result on the web page
        return render_template('index.html', prediction=prediction_result)

@app.route('/error/<error_message>')
def error(error_message):
    return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
