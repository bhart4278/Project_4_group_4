# Project 4

Tableau Deck: https://public.tableau.com/app/profile/oana.wright/viz/Furr-ever_friend/ProjectOverview?publish=yes

# Dog Breed Recommendation Model

## Group 4 Members
Oana Wright, Brian Hart, Rogelio Cardenas, Adam Butcher

## Project Overview
This project is a web-based application designed to recommend the best dog breed group based on user-selected characteristics. We leverage machine learning models, specifically **Random Forest Classifier** and **XGBoost**, to provide accurate predictions. The model is trained using 235 distinct dog breeds.

![Screenshot 2025-03-11 at 6 38 36 PM](https://github.com/user-attachments/assets/bb707bee-fb03-4106-84ce-1458e15a977f)


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

#### XGBoost improves performance by building new trees based on the residuals (errors) of previous trees, rather than training each tree independently. This iterative process, known as gradient boosting, allows each new tree to correct the mistakes made by the earlier trees. By adding these corrections, XGBoost reduces bias and variance, leading to a more accurate and robust model that can better capture complex relationships in the data.

## Visualizations

### Data Statistics by Category
- Displays the most significant factors influencing breed group recommendations.
- Users can select which breed characteristics chart they want to see by using the drop down.
- Helps users understand which traits impact the predictions the most.

![Screenshot 2025-03-11 at 7 33 39 PM](https://github.com/user-attachments/assets/bc2bc1d9-3fad-42b1-a79b-dc0bee681bc1)

### Model Performance Metrics
- Accuracy scores for **Random Forest** and **XGBoost** are displayed.
- Training time and prediction response times are recorded.

![Screenshot 2025-03-11 at 7 31 53 PM](https://github.com/user-attachments/assets/5ac41aea-e28a-4a2f-abed-e1cdbb6d0b18)

## Additional Features

### User Input Filtering
- Users can select desired traits using dropdown menus.
- Input values are dynamically processed before making predictions.

### Interactive Results Display
- Predictions are displayed alongside breed group characteristics.
- Accuracy and processing times are shown for transparency.

![Screenshot 2025-03-11 at 7 54 58 PM](https://github.com/user-attachments/assets/322e3392-01d3-4772-b48d-e997eaf161d5)


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

## References
This project was completed as part of the **University of Texas at Austin, McCombs Business School Data Visualization and Analysis Bootcamp.**

Data Source: https://www.kaggle.com/datasets/mexwell/dog-breeds-dataset?resource=download

Dog Images: https://dogell.com/en

---
Enjoy! 🐶

