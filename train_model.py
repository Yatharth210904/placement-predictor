import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# 1. GENERATE DATASET
np.random.seed(42)
num_samples = 300
data = {
    'cgpa': np.random.uniform(5.5, 9.8, num_samples),
    'aptitude_score': np.random.randint(45, 98, num_samples),
    'internships': np.random.randint(0, 4, num_samples),
    'backlogs': np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05], size=num_samples),
    'stream': np.random.choice(['CS', 'IT', 'ECE', 'EE'], size=num_samples),
    'work_exp': np.random.choice(['Yes', 'No'], p=[0.3, 0.7], size=num_samples)
}
df = pd.DataFrame(data)

placement_chance = (df['cgpa'] * 0.4) + (df['aptitude_score'] * 0.005) + (df['internships'] * 0.1) - (df['backlogs'] * 0.15)
df['status'] = np.where(placement_chance > 3.4, 'Placed', 'Not Placed')
df['salary'] = np.where(df['status'] == 'Placed', np.round(df['cgpa'] * 0.8 + df['internships'] * 0.5 + np.random.uniform(1.5, 3.5, num_samples), 2), 0.0)

# 2. DATA PREPROCESSING
le_stream = LabelEncoder()
df['stream'] = le_stream.fit_transform(df['stream'])
le_work_exp = LabelEncoder()
df['work_exp'] = le_work_exp.fit_transform(df['work_exp'])

X = df[['cgpa', 'aptitude_score', 'internships', 'backlogs', 'stream', 'work_exp']]
y_class = df['status']

X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. TRAINING MODEL
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_scaled, y_train_class)

placed_filter_train = (y_train_class == 'Placed')
X_train_reg = X_train_scaled[placed_filter_train]
y_train_reg = df.loc[X_train.index[placed_filter_train], 'salary']

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

# 4. SAVE ARTIFACTS
joblib.dump(clf_model, 'placement_classifier.pkl')
joblib.dump(reg_model, 'salary_regressor.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(le_stream, 'stream_encoder.pkl')
joblib.dump(le_work_exp, 'work_exp_encoder.pkl')

print("Success: Models trained and saved!")