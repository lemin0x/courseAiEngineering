import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Prepare data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(model, 'ai_model.pkl')

# Load and predict
loaded_model = joblib.load('ai_model.pkl')
new_data = np.random.randn(1, 3)
prediction = loaded_model.predict(new_data)
print(f"Prediction for new data: {prediction[0]}")