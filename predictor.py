import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
data = pd.read_csv("C:\\Users\\kiran\\Desktop\\NEWSS1.csv")

# Drop columns with no observed values
data = data.dropna(axis=1, how='all')

# Split the data into features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')  
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets after imputation
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

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
