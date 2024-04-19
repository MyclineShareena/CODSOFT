"""
#Importing the Dependencies
"""

import numpy as np
#data preprocessing
import pandas as pd
#data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#train_test_split
from sklearn.model_selection import train_test_split
#MLmodel
from sklearn.linear_model import LogisticRegression
#evaluationmetrics
from sklearn.metrics import accuracy_score

"""# Data Collection and Processing"""

#Load the data from csv file to pandas dataframe
titanic_data = pd.read_csv("tested.csv")

#print first 5rows of dataframe
titanic_data.head()

#number of rows and columns
titanic_data.shape

#getting some information from data
titanic_data.info()

#check the number of missing values in each column
titanic_data.isnull().sum()

"""Handling Missing values"""

#drop the cabin column from dataframe
titanic_data = titanic_data.drop(columns="Cabin",axis=1)

#replacing the missing values in Age & Fare column with mean values
titanic_data["Age"].fillna(titanic_data["Age"].mean, inplace=True)
titanic_data["Fare"].fillna(titanic_data["Fare"].mean, inplace=True)

titanic_data.isnull().sum()

"""# Data Analysis"""

#getting some stastical measures about data
titanic_data.describe()

#finding the number of people survived and not survived
titanic_data["Survived"].value_counts()

"""# Data Visualization"""

#theme
sns.set()

#making a counnt plot for "Survived" column
sns.countplot(x="Survived", data=titanic_data)

#finding the number of people survived and not survived based on sex
titanic_data["Sex"].value_counts()

#making a counnt plot for "Sex" column
sns.countplot(x="Sex", data=titanic_data)

#number of survivors based on gender
sns.countplot(x="Sex",hue="Survived",data=titanic_data)

#making a counnt plot for "Pclass" column
sns.countplot(x="Pclass", data=titanic_data)

#number of survivors based on Pclass
sns.countplot(x="Pclass",hue="Survived",data=titanic_data)

# Encoding categorical variables
titanic_data.replace({'Sex' : {'female': 1, 'male': 0}, 'Embarked' : {'S': 0, 'C': 1, 'Q': 2}}, inplace= True)

"""Seperating Features and Target"""

X = titanic_data.drop(columns= ['PassengerId','Name','Ticket','Survived'], axis=1)
Y = titanic_data["Survived"]

"""# Splitting the data into Train and Test"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

"""# Model Training"""

LR = LogisticRegression(solver='liblinear', max_iter=200)
LR.fit(X_train, Y_train)
Y_pred = LR.predict(X_test)

"""# Model Evaluation

"""

LRAcc = accuracy_score(Y_pred, Y_test)
print('Logistic regression accuracy: {:.2f}%'.format(LRAcc * 100))