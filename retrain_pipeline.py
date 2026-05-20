import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os

print("--- Starting Model Upgrade Pipeline ---")

# 1. GENERATE / LOAD BASE DATASET
np.random.seed(42)
num_samples = 300
base_data = {
    'cgpa': np.random.uniform(5.5, 9.8, num_samples),
    'aptitude_score': np.random.randint(45, 98, num_samples),
    'internships': np.random.randint(0, 4, num_samples),
    'backlogs': np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05], size=num_samples),
    'stream': np.random.choice(['CS', 'IT', 'ECE', 'EE'], size=num_samples),
    'work_exp': np.random.choice(['Yes', 'No'], p=[0.3, 0.7], size=num_samples)
}
df_base = pd.DataFrame(base_data)

placement_chance = (df_base['cgpa'] * 0.4) + (df_base['aptitude_score'] * 0.005) + (df_base['internships'] * 0.1) - (df_base['backlogs'] * 0.15)
df_base['status'] = np.where(placement_chance > 3.4, 'Placed', 'Not Placed')
df_base['salary'] = np.where(df_base['status'] == 'Placed', np.round(df_base['cgpa'] * 0.8 + df_base['internships'] * 0.5 + np.random.uniform(1.5, 3.5, num_samples), 2), 0.0)

# 2. CHECK AND MERGE COLLECTED USER DATA
user_data_path = 'collected_user_data.csv'

if os.path.exists(user_data_path):
    print(f"Found user logs in '{user_data_path}'. Merging datasets...")
    df_user = pd.read_csv(user_data_path)
    
    if not df_user.empty:
        # Match data format cleanly
        df_user_cleaned = df_user[['cgpa', 'aptitude_score', 'internships', 'backlogs', 'stream', 'work_exp', 'status', 'salary']]
        df_final = pd.concat([df_base, df_user_cleaned], ignore_index=True)
        print(f"Dataset expanded! Combined rows count: {len(df_final)}")
    else:
        df_final = df_base
        print("User log file is empty. Proceeding with base data.")
else:
    df_final = df_base
    print("No user data logs found yet. Retraining models purely on base data.")

# 3. DATA PREPROCESSING PIPELINE
le_stream = LabelEncoder()
df_final['stream'] = le_stream.fit_transform(df_final['stream'])

le_work_exp = LabelEncoder()
df_final['work_exp'] = le_work_exp.fit_transform(df_final['work_exp'])

X = df_final[['cgpa', 'aptitude_score', 'internships', 'backlogs', 'stream', 'work_exp']]
y_class = df_final['status']

X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. RETRAIN MODULES
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_scaled, y_train_class)

placed_filter_train = (y_train_class == 'Placed')
X_train_reg = X_train_scaled[placed_filter_train]
y_train_reg = df_final.loc[X_train.index[placed_filter_train], 'salary']

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

# 5. OVERWRITE OLD MODELS
joblib.dump(clf_model, 'placement_classifier.pkl')
joblib.dump(reg_model, 'salary_regressor.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(le_stream, 'stream_encoder.pkl')
joblib.dump(le_work_exp, 'work_exp_encoder.pkl')

print("Success: Models successfully upgraded with fresh user interactions!")