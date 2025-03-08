import time
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform

app = Flask(__name__)

# Load and preprocess data
file_path = "Resources/oana-akc-data.csv"
data = pd.read_csv(file_path)

data = data.rename(columns={
    'grooming_frequency_category': 'grooming',
    'shedding_category': 'shedding',
    'energy_level_category': "energy_level",
    'trainability_category': 'trainability',
    "demeanor_category": "demeanor"
})

data = data.drop(columns=["Dog Breed", "description", "temperament", "grooming_frequency_value",
                          "shedding_value", "energy_level_value", "trainability_value", "demeanor_value"])

# Encode categorical target variable
y = data["group"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode categorical features
label_encoders = {}
categorical_columns = ['grooming', 'shedding', 'energy_level', 'trainability', 'demeanor']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

X = data.drop(columns=["group"])

# Random Forest RandomizedSearchCV
rf_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# XGBoost RandomizedSearchCV
xgb_param_dist = {
    'n_estimators': randint(50, 150),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 7),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'scale_pos_weight': randint(1, 10)
}

# Random Forest RandomizedSearchCV
rf_random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=rf_param_dist, 
                                      n_iter=10, cv=3, verbose=2, n_jobs=-1, random_state=42, scoring='accuracy')

# XGBoost RandomizedSearchCV
xgb_random_search = RandomizedSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), 
                                       param_distributions=xgb_param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1, 
                                       random_state=42, scoring='accuracy')

# Train Random Forest with RandomizedSearchCV
start_train_time = time.time()
rf_random_search.fit(X, y_encoded)
end_train_time = time.time()
training_time_rf = end_train_time - start_train_time
best_rf_model = rf_random_search.best_estimator_

# Train XGBoost with RandomizedSearchCV
start_train_time = time.time()
xgb_random_search.fit(X, y_encoded)
end_train_time = time.time()
training_time_xgb = end_train_time - start_train_time
best_xgb_model = xgb_random_search.best_estimator_

# Accuracy of both models
accuracy_rf = accuracy_score(y_encoded, best_rf_model.predict(X))
accuracy_xgb = accuracy_score(y_encoded, best_xgb_model.predict(X))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        demeanor = request.form['demeanor']
        energy_level = request.form['energy_level']
        trainability = request.form['trainability']
        grooming = request.form['grooming']
        shedding = request.form['shedding']
        min_height = request.form['min_height']
        max_height = request.form['max_height']
        min_weight = request.form['min_weight']
        max_weight = request.form['max_weight']
        min_expectancy = request.form['min_expectancy']
        max_expectancy = request.form['max_expectancy']

        # Process the form values and predict the dog breed
        start_api_time = time.time()

        user_input = {
            'demeanor': demeanor,
            'energy_level': energy_level,
            'trainability': trainability,
            'grooming': grooming,
            'shedding': shedding,
            'min_height': float(min_height),
            'max_height': float(max_height),
            'min_weight': float(min_weight),
            'max_weight': float(max_weight),
            'min_expectancy': float(min_expectancy),
            'max_expectancy': float(max_expectancy)
        }

        # Convert user input to encoded format
        user_vector = []
        for col in X.columns:
            if col in label_encoders:
                try:
                    encoded_value = label_encoders[col].transform([user_input[col]])[0]
                except ValueError:
                    encoded_value = -1  # Handle unseen categories
            else:
                encoded_value = user_input[col]
            user_vector.append(encoded_value)
        
        user_df = pd.DataFrame([user_vector], columns=X.columns)

        # XGBoost prediction
        start_pred_time = time.time()
        pred_xgb = best_xgb_model.predict(user_df)[0]
        pred_xgb_label = label_encoder.inverse_transform([pred_xgb])[0]
        end_pred_time = time.time()
        prediction_time_xgb = end_pred_time - start_pred_time

        # RandomForest prediction
        start_pred_time = time.time()
        pred_rf = best_rf_model.predict(user_df)[0]
        pred_rf_label = label_encoder.inverse_transform([pred_rf])[0]
        end_pred_time = time.time()
        prediction_time_rf = end_pred_time - start_pred_time

        end_api_time = time.time()
        api_response_time = end_api_time - start_api_time

        return render_template("index.html",
                               demeanor=demeanor,
                               energy_level=energy_level,
                               trainability=trainability,
                               grooming=grooming,
                               shedding=shedding,
                               min_height=min_height,
                               max_height=max_height,
                               min_weight=min_weight,
                               max_weight=max_weight,
                               min_expectancy=min_expectancy,
                               max_expectancy=max_expectancy,
                               prediction_text_xgb=f"XGBoost: {pred_xgb_label}",
                               prediction_text_rf=f"RandomForest: {pred_rf_label}",
                               training_time_xgb=f"{training_time_xgb:.4f} sec",
                               training_time_rf=f"{training_time_rf:.4f} sec",
                               prediction_time_xgb=f"{prediction_time_xgb:.6f} sec",
                               prediction_time_rf=f"{prediction_time_rf:.6f} sec",
                               api_response_time=f"{api_response_time:.4f} sec",
                               model_accuracy_xgb=f"XGBoost Accuracy: {accuracy_xgb * 100:.2f}%",
                               model_accuracy_rf=f"RandomForest Accuracy: {accuracy_rf * 100:.2f}%")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
