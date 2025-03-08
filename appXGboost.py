import time
import pandas as pd
import pickle
import numpy as np
from flask import Flask, request, render_template
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score  # Import accuracy_score

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
file_path = "Resources/oana-akc-data.csv"
data = pd.read_csv(file_path)

data = data.drop(columns=["Dog Breed", "description", "temperament"])

# Separate features and target variable
X = data.drop(columns=["group"])  # Features
y = data["group"]  # Target variable

# Convert all object-type feature columns to categorical
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

# Fit LabelEncoder on the full target since each breed appears only once.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Unique values in encoded labels: {set(y_encoded)}")
print(f"Expected labels (continuous): {set(range(len(label_encoder.classes_)))}")

# Since every breed appears only once, we use the full dataset for training.
X_train = X
y_train = y_encoded

# ---------------------------
# Train the Model
# ---------------------------
start_train_time = time.time()

xgb_model = XGBClassifier(n_estimators=100,
                          learning_rate=0.1,
                          use_label_encoder=False,
                          eval_metric='mlogloss',
                          enable_categorical=True)
xgb_model.fit(X_train, y_train)

end_train_time = time.time()
training_time = end_train_time - start_train_time
print(f"✅ Training completed in {training_time:.2f} seconds")

# Calculate accuracy on training data
y_train_pred = xgb_model.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy

print(f"✅ Model accuracy on training data: {accuracy * 100:.2f}%")

# Save the trained model
with open("xgb_model.pkl", "wb") as model_file:
    pickle.dump(xgb_model, model_file)
print("✅ Model saved as 'xgb_model.pkl'")

# ---------------------------
# Save Category Information
# ---------------------------
# Record which columns are categorical and their categories from training data.
categorical_cols = X.select_dtypes(include='category').columns.tolist()
cat_categories = {col: X[col].cat.categories for col in categorical_cols}

# ---------------------------
# Flask API
# ---------------------------
with open("xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_breed():
    if request.method == "POST":
        start_api_time = time.time()
        
        # Build dictionary from form data.
        # Try converting each value to float; if it fails, keep as string.
        user_input = {}
        for key, value in request.form.items():
            try:
                user_input[key] = float(value)
            except ValueError:
                user_input[key] = value
        
        # Create DataFrame from the user input.
        input_df = pd.DataFrame([user_input])
        # Ensure input_df has all feature columns as in training (fill missing ones with NaN)
        input_df = input_df.reindex(columns=X_train.columns, fill_value=np.nan)
        
        # For each categorical column, convert the column to a categorical dtype using the same categories as training.
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = pd.Categorical(input_df[col], categories=cat_categories[col])
        
        # Make prediction and decode breed label.
        start_pred_time = time.time()
        prediction = model.predict(input_df)[0]
        prediction_breed = label_encoder.inverse_transform([prediction])[0]
        end_pred_time = time.time()
        prediction_time = end_pred_time - start_pred_time
        
        end_api_time = time.time()
        api_response_time = end_api_time - start_api_time
        
        # Return the prediction result along with model accuracy
        return render_template('index.html', 
                               prediction_text=f"The best-matching dog breed for you is: {prediction_breed}",
                               training_time=f"{training_time:.4f} sec",
                               prediction_time=f"{prediction_time:.6f} sec",
                               api_response_time=f"{api_response_time:.4f} sec",
                               model_accuracy=f"Model Accuracy: {accuracy * 100:.2f}%")  # Add accuracy here
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
