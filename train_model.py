import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('Training.csv')

# Features and target
X = data.drop(columns=['prognosis'])
y = data['prognosis']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model
with open('disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)
