import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the clean dataset
print("Loading dataset...")
df = pd.read_csv('training_data_final.csv')

# 2. Extract Features
X = df[['Voltage', 'Current', 'Frequency', 'ActivePower']]
y = df['AnomalyFlag']

# 3. Filter for ONLY normal baseline operations (y == 0)
X_normal = X[y == 0]
print(f"Training Isolation Forest on {len(X_normal)} normal grid records...")

# 4. Train the Isolation Forest model
iso_forest = IsolationForest(
    n_estimators=100,
    contamination='auto',  # Automatically estimates boundary threshold
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_normal)

# 5. Evaluate on the entire dataset (Inliers: 1, Outliers: -1)
print("\nEvaluating on full dataset...")
raw_preds = iso_forest.predict(X)

# Convert outputs to match 0 (Normal) and 1 (Anomaly)
iso_preds = [1 if p == -1 else 0 for p in raw_preds]

print("\n--- Isolation Forest Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y, iso_preds))
print("\nClassification Report:\n", classification_report(y, iso_preds, digits=4))

# 6. Save Model to Disk
model_filename = 'iso_forest_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(iso_forest, f)

print(f"\nModel successfully saved as '{model_filename}'!")