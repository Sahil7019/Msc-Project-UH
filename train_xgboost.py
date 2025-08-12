import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# === File path to your dataset ===
file_path = '/home/kali/Desktop/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

# === Load the dataset ===
data = pd.read_csv(file_path)

# === Strip leading/trailing whitespace from all column names ===
data.columns = data.columns.str.strip()

# === Replace inf/-inf with NaN and drop any remaining NaNs ===
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# === Drop constant columns ===
data = data.loc[:, data.nunique() > 1]

# === Drop non-numeric columns except 'Label' ===
non_numeric_cols = data.select_dtypes(include=['object']).columns
non_label_cols = [col for col in non_numeric_cols if col != 'Label']
data.drop(columns=non_label_cols, inplace=True)

# === Encode the Label: 'BENIGN' -> 0, others -> 1 ===
data['Label'] = data['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

# === Split into features and target ===
X = data.drop('Label', axis=1)
y = data['Label']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Train the XGBoost model ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=1)
model.fit(X_train, y_train)

# === Evaluate the model ===
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))

# === Save the model to a file ===
joblib.dump(model, 'xgboost_ids_model.pkl')
print("\n✅ Model saved successfully to 'xgboost_ids_model.pkl'")
