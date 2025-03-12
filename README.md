# Project_4_group_4
Final Project for UT Data Visualization and Analysis

# Dog Breed Recommendation Model

## Group Members
Oana Wright, Brian Hart, Rogelio Cardenas, Adam Butcher

## Project Overview
This project is a web-based application designed to recommend the best dog breed group based on user-selected characteristics. It leverages machine learning models, specifically **Random Forest Classifier** and **XGBoost**, to provide accurate predictions.

## How to Use the Application
- Users input their preferences for dog traits such as energy level, grooming needs, trainability, and demeanor.
- The application processes these inputs and predicts the best-matching dog breed group.
- Results are displayed along with model accuracy and response times.

## Data Processing Workflow

### Fetching and Processing Data
- The raw dataset is loaded from a CSV file and stored in an SQLite database.
- The data is cleaned by selecting relevant columns and removing missing values.
- The cleaned dataset is saved for further processing.

### Model Training and Optimization
- Categorical features are encoded for machine learning.
- **RandomizedSearchCV** is used for hyperparameter tuning of Random Forest and XGBoost models.
- The models are trained, and accuracy scores are evaluated.

## Visualizations

### Feature Importance Chart
- Displays the most significant factors influencing breed group recommendations.
- Helps users understand which traits impact the predictions the most.

### Model Performance Metrics
- Accuracy scores for **Random Forest** and **XGBoost** are displayed.
- Training time and prediction response times are recorded.

## Additional Features

### User Input Filtering
- Users can select desired traits using dropdown menus.
- Input values are dynamically processed before making predictions.

### Interactive Results Display
- Predictions are displayed alongside breed group characteristics.
- Accuracy and processing times are shown for transparency.

## Technology Stack

### Data Processing:
- **Python**
- **Pandas**
- **SQLite**

### Machine Learning:
- **Scikit-Learn**
- **XGBoost**
- **Random Forest Classifier**

### Web Application:
- **Flask**
- **HTML/CSS**
- **Jinja2** (for dynamic content rendering) did we use this?

## Project URL
TBD (Deploy Link if applicable)

## References
Data Source: https://www.kaggle.com/datasets/mexwell/dog-breeds-dataset?resource=download
Dog Images: https://dogell.com/en#google_vignette
---
Enjoy! 🐶

