# predictor.py

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Split the data into features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print the accuracy on the test set
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    # Save the trained model
    joblib.dump(model, 'banking_crisis_model.joblib')

def load_and_predict(input_data, model_path='banking_crisis_model.joblib'):
    # Load the model from the file
    loaded_model = joblib.load(model_path)

    # Make predictions on the input data
    predictions = loaded_model.predict(input_data)

    return predictions

# Example usage
if __name__ == "__main__":
    # Train and save the model (you may call this only once)
    train_and_save_model("C:\\Users\\kiran\\Desktop\\gb.csv")

    # Example input data for prediction (replace this with your actual input)
    input_data = ...

    # Make predictions using the loaded model
    result = load_and_predict(input_data)

    # Print or use the result as needed
    print(result)
