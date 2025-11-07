# Classification_Project
Titanic Survival Prediction.  This project is a machine learning pipeline that predicts the survival of passengers on the Titanic using the classic Kaggle dataset. It covers the full workflow from data exploration to model evaluation and prediction.

Project Overview
The dataset contains information about passengers, including demographic details (age, sex), travel details (ticket class, fare), and family information (siblings/spouses, parents/children). The goal is to predict the Survived column, indicating whether a passenger survived or not.

Features & Preprocessing
The project involves data cleaning, feature engineering, and encoding, including:
Handling missing values in Age, Fare, and Embarked.
Dropping irrelevant columns like Cabin, PassengerId, Name, and Ticket.
Creating new features:
Family – sum of SibSp and Parch to represent family size on board.
Ticket_group_size – number of passengers sharing the same ticket.
Categorizing Age into groups (Infant, Child, Teenager, Young Adult, Adult, Senior).
One-hot encoding categorical variables (Sex, Embarked, Age_categories, Pclass).

Exploratory Data Analysis (EDA)
Visualizations and statistical analysis were used to identify patterns:
Survival rates by gender (Sex) and passenger class (Pclass).
Age distributions within each passenger class.
Ticket group sizes and their distribution among passengers.
Embarkation points and their correlation with survival and class.

Model Building
The project implements Logistic Regression as the predictive model:
Data is split into training and test sets.
Standard scaling is applied to numeric features (Age, Fare, Ticket_group_size).
Model evaluation using:
Classification report (precision, recall, F1-score).
10-fold cross-validation to assess stability.

Prediction on Test Set
The test dataset undergoes the same preprocessing steps, including missing value imputation, feature engineering, and scaling. The trained Logistic Regression model predicts survival outcomes (Survived) for all passengers in the test set.

Technologies Used
Python
Pandas & NumPy (data manipulation)
Matplotlib & Seaborn (visualization)
Scikit-learn (modeling, preprocessing, evaluation)

Usage
Clone the repository.
Install required packages: pip install pandas numpy matplotlib seaborn scikit-learn.
Run the notebook/script with the provided train.csv and test.csv datasets.
Obtain survival predictions and evaluate model performance.
