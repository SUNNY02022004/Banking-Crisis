from flask import Flask, render_template, request
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
        # Extract features from the form
        features = [float(request.form[f'feature_{i}']) for i in range(10)]  # Update the range to 10
        
        # Create a DataFrame from the input data with correct column names
        input_data = pd.DataFrame([features], columns=feature_names)
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)[0]
        
        # Render the prediction result on the web page
        return render_template('index.html', prediction=prediction)

#if __name__ == '__main__':
    #app.run(debug=True, use_reloader=False)
