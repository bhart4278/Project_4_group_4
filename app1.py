import time
from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score

app = Flask(__name__)

# Load data and preprocess (same as your existing code)
file_path = "Resources/oana-akc-data.csv"
data = pd.read_csv(file_path)
data = data.rename(columns={'grooming_frequency_category': 'grooming', 
                             'shedding_category': 'shedding', 
                             'energy_level_category':"energy_level",
                             'trainability_category':'trainability',
                             "demeanor_category":"demeanor"})

data = data.drop(columns=["Dog Breed", "description", "temperament","grooming_frequency_value","shedding_value","energy_level_value","trainability_value","demeanor_value"])

# Define label encoders and fit them on all possible values
demeanor_values = ["Aloof/Wary", "Reserved with Strangers", "Alert/Responsive", "Friendly", "Outgoing"]
energy_values = ["Couch Potato", "Calm", "Energetic", "Regular Exercise", "Needs Lots of Activity"]
trainability_values = ["May be Stubborn", "Independent", "Easy Training", "Agreeable", "Eager to Please"]
grooming_values = ["Daily Brushing", "2-3 Times a Week Brushing", "Weekly Brushing", "Occasional Bath/Brush", "Specialty/Professional"]
shedding_values = ["Infrequent", "Occasional", "Seasonal", "Regularly", "Frequent"]

label_encoders = {
    'demeanor': LabelEncoder().fit(demeanor_values),
    'energy_level': LabelEncoder().fit(energy_values),
    'trainability': LabelEncoder().fit(trainability_values),
    'grooming': LabelEncoder().fit(grooming_values),
    'shedding': LabelEncoder().fit(shedding_values),
}

# Encode categorical variables
for col in data.columns[1:]:
    if data[col].dtype == 'object':  
        try:
            data[col] = label_encoders[col].transform(data[col])
        except KeyError:
            print(f"Unseen category found in column {col}")

# Define features and target variable
X = data.drop(columns=["group"])
y = data["group"]

# Measure training time
start_train_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(X, y)
end_train_time = time.time()
training_time = end_train_time - start_train_time  # Store training time

# Calculate accuracy on training data
y_train_pred = rf_model.predict(X)
accuracy = accuracy_score(y, y_train_pred)  # Calculate accuracy

print(f"✅ Model accuracy on training data: {accuracy * 100:.2f}%")

@app.route('/', methods=['GET', 'POST'])
def predict_breed():
    if request.method == 'POST':
        start_api_time = time.time()  # Start API timing

        user_input = {
            "grooming": request.form["grooming"],
            "shedding": request.form["shedding"],
            "energy_level": request.form["energy_level"],
            "trainability": request.form["trainability"],
            "demeanor": request.form["Demeanor"],
            "min_height": float(request.form["min_height"]),
            "max_height": float(request.form["max_height"]),
            "min_weight": float(request.form["min_weight"]),
            "max_weight": float(request.form["max_weight"]),
            "min_expectancy": float(request.form["min_expectancy"]),
            "max_expectancy": float(request.form["max_expectancy"])
        }
        
        user_vector = []
        for col, value in user_input.items():
            if col in label_encoders:
                try:
                    encoded_value = label_encoders[col].transform([value])[0]
                except KeyError:
                    encoded_value = -1  # Default for unseen values
            else:
                encoded_value = value
            user_vector.append(encoded_value)
        
        user_df = pd.DataFrame([user_vector], columns=X.columns)

        start_pred_time = time.time()  # Start prediction timing
        predicted_breed = rf_model.predict(user_df)[0]
        end_pred_time = time.time()  # End prediction timing

        prediction_time = end_pred_time - start_pred_time  # Store prediction time
        end_api_time = time.time()  # End API timing
        api_response_time = end_api_time - start_api_time  # Store API response time

        return render_template('index.html', 
                               prediction_text=f"The best-matching dog breed for you is: {predicted_breed}",
                               training_time=f"{training_time:.4f} sec",
                               prediction_time=f"{prediction_time:.6f} sec",
                               api_response_time=f"{api_response_time:.4f} sec",
                               model_accuracy=f"Model Accuracy: {accuracy * 100:.2f}%")  # Add accuracy here

    return render_template('index.html', prediction_text=None, training_time=None, prediction_time=None, api_response_time=None, model_accuracy=None)

if __name__ == '__main__':
    app.run(debug=True)
