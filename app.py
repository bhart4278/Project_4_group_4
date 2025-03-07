from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the cleaned breed data
file_path = "Resources/akc-data-cleaned_2.csv"
data = pd.read_csv(file_path)

# Change column names
data = data.rename(columns={'grooming_frequency_category': 'grooming', 
                             'shedding_category': 'shedding', 
                             'energy_level_category':"energy_level",
                             'trainability_category':'trainability',
                             "demeanor_category":"demeanor"})


# Define all possible categories for the label encoders
demeanor_values = ["Aloof/Wary", "Reserved with Strangers", "Alert/Responsive", "Friendly", "Outgoing"]
energy_values = ["Couch Potato", "Calm", "Energetic", "Regular Exercise", "Needs Lots of Activity"]
trainability_values = ["May be Stubborn", "Independent", "Easy Training", "Agreeable", "Eager to Please"]
grooming_values = ["Daily Brushing", "2-3 Times a Week Brushing", "Weekly Brushing", "Occasional Bath/Brush", "Specialty/Professional"]
shedding_values = ["Infrequent", "Occasional", "Seasonal", "Regularly", "Frequent"]

# Initialize the LabelEncoders and fit them on all possible values
label_encoders = {}
label_encoders['demeanor'] = LabelEncoder()
label_encoders['demeanor'].fit(demeanor_values)

label_encoders['energy_level'] = LabelEncoder()
label_encoders['energy_level'].fit(energy_values)

label_encoders['trainability'] = LabelEncoder()
label_encoders['trainability'].fit(trainability_values)

label_encoders['grooming'] = LabelEncoder()
label_encoders['grooming'].fit(grooming_values)

label_encoders['shedding'] = LabelEncoder()
label_encoders['shedding'].fit(shedding_values)

# Encode categorical variables
for col in data.columns[1:]:
    if data[col].dtype == 'object':  # Only encode categorical columns
        try:
            data[col] = label_encoders[col].transform(data[col])
        except KeyError:
            print(f"Unseen category found in column {col}")

# Define features and target variable
X = data.drop(columns=["Dog Breed"])  # Features
y = data["Dog Breed"]  # Target variable (Dog Breed)

# Train Random Forest Classifier (this happens once at the beginning)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def predict_breed():
    if request.method == 'POST':
        user_input = {
            "grooming": request.form["grooming"],
            "shedding": request.form["shedding"],
            "energy_level": request.form["energy_level"],
            "trainability": request.form["trainability"],
            "demeanor": request.form["Demeanor"],
            # Numerical inputs
            "min_height": float(request.form["min_height"]),
            "max_height": float(request.form["max_height"]),
            "min_weight": float(request.form["min_weight"]),
            "max_weight": float(request.form["max_weight"]),
            "min_expectancy": float(request.form["min_expectancy"]),
            "max_expectancy": float(request.form["max_expectancy"])
        }
        
        # Encode categorical user input
        user_vector = []
        for col, value in user_input.items():
            if col in label_encoders:  # If it's a categorical feature
                try:
                    encoded_value = label_encoders[col].transform([value])[0]
                except KeyError:
                    print(f"Unseen value for {col}: {value}")
                    encoded_value = -1  # Default for unseen values
            else:
                # If it's a numerical value, just add it as is
                encoded_value = value
            user_vector.append(encoded_value)
        
        # Convert input into a DataFrame for prediction
        user_df = pd.DataFrame([user_vector], columns=X.columns)
        
        # Predict the best dog breed
        predicted_breed = rf_model.predict(user_df)[0]
        
        return render_template('index.html', prediction_text=f"The best-matching dog breed for you is: {predicted_breed}")
    
    return render_template('index.html', prediction_text=None)

if __name__ == '__main__':
    app.run(debug=True)
