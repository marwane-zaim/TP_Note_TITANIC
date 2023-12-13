import pandas as pd
import numpy as np
from explore import import_data_Train

#Load data file
data_train = import_data_Train()
# data_test = import_data_Test()

def transformation(data):
    data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1, inplace=True)
    data['Sex'] = data['Sex'].replace({'male': 1, 'female': 0})
    print(data.head())

    missing_values = data.isnull().sum()
    print("Missing values before filling:", missing_values)

    data['Age'].fillna(data['Age'].mean(), inplace=True)
    
    missing_values_after_fill = data['Age'].isnull().sum()
    
    print("Missing values after filling:", missing_values_after_fill)
    
    print("Filled 'Age' column:", data['Age'])
    

    return data

transformation(data_train)