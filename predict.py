import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Step 1: Generating Synthetic Sensor Data...")
np.random.seed(42)
num_samples = 5000

# Generate Normal Operation Data (No Leak)
normal_flow = np.random.normal(50, 5, 4000)      # Around 50 L/min
normal_pressure = np.random.normal(300, 20, 4000) # Around 300 kPa
normal_moisture = np.random.normal(20, 5, 4000)   # Around 20%
normal_labels = np.zeros(4000)                    # 0 = No Leak

# Generate Leak Data (Anomalies)
leak_flow = np.random.normal(30, 8, 1000)         # Flow drops
leak_pressure = np.random.normal(200, 30, 1000)   # Pressure drops
leak_moisture = np.random.normal(60, 15, 1000)    # Moisture spikes
leak_labels = np.ones(1000)                       # 1 = Leak

# Combine into a DataFrame
flow = np.concatenate([normal_flow, leak_flow])
pressure = np.concatenate([normal_pressure, leak_pressure])
moisture = np.concatenate([normal_moisture, leak_moisture])
labels = np.concatenate([normal_labels, leak_labels])

df = pd.DataFrame({
    'Flow_Rate': flow,
    'Pressure': pressure,
    'Soil_Moisture': moisture,
    'Leak_Detected': labels
})

print("Step 2: Training the Random Forest Model...")
# Split data into features (X) and target (y)
X = df[['Flow_Rate', 'Pressure', 'Soil_Moisture']]
y = df['Leak_Detected']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("Step 3: Evaluating the Model...")
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, predictions))

print("Step 4: Saving the Model for Deployment...")
# Save the trained model to a file
joblib.dump(model, 'water_leak_model.pkl')
print("Model successfully saved as 'water_leak_model.pkl'!")