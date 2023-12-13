import pandas as pd
import numpy as np
from explore import import_data_Train

#Load data file
data_train = import_data_Train()
# data_test = import_data_Test()

def transformation(data):
    #Apply the clean function to the ticket column
    data['Ticket'] = data['Ticket'].apply(clean_ticket)
    print(data['Ticket'].head(30))

    missing_values = data.isnull().sum()
    print("Missing values before filling:", missing_values)

    data['Age'].fillna(data['Age'].mean(), inplace=True)
    
    data['Cabin'].fillna('C00', inplace=True)
    
    data['Cabin'] = data['Cabin'].str[:4].str.replace(r'\s', '')
    
    missing_values_after_fill = data['Age'].isnull().sum()
    
    print("Missing values after filling:", missing_values_after_fill)
    
    print("Filled 'Age' column:", data['Age'])
    
    print("Filled 'Cabin' column:", data['Cabin'].head(30))

    options = ["S", "C", "Q"]
    data['Embarked'].fillna(np.random.choice(options), inplace=True)

    return data    

#function to clean the ticket column
def clean_ticket(ticket):
    if ' ' in ticket:
        return ticket.split()[-1]
    else:
        return ticket

transformation(data_train)