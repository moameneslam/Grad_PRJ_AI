import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the decoded dataset
print("Loading dataset...")
df = pd.read_csv('training_data_final.csv')

# 2. Extract Features and Target
X = df[['Voltage', 'Current', 'Frequency', 'ActivePower']]
y = df['AnomalyFlag']

# 3. Stratified Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train XGBoost Classifier
print("Training XGBoost Classifier...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='binary:logistic',
    random_state=42
)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# 6. Save Model Weights
model.save_model('xgboost_fault_classifier.json')
print("\nTrained model saved as 'xgboost_fault_classifier.json'")