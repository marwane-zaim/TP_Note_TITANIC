import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load data file
def import_data_Train() -> pd.DataFrame:
    """
    Import csv file as a dataframe
    Output: data [pd.DataFrame]
    """
    data = pd.read_csv("Data/train.csv")
    return data

# def import_data_Test() -> pd.DataFrame:
#     """
#     Import csv file as a dataframe
#     Output: data [pd.DataFrame]
#     """
#     data = pd.read_csv("Data/test.csv")
#     return data

data = import_data_Train()

# #Show the first rows to understand the data structure
# print(data.head())

# #information about data tyypes, missing values, etc
# print(data.info())

# #
# print(data.describe())
# print(data['Survived'].value_counts())

# sns.pairplot(data, hue='Survived')
# plt.show()

# sns.countplot(x='Survived', data=data)
# plt.title('Repartition des survivants')
# plt.show()

# sns.countplot(x='Survived', hue='Sex', data=data)
# plt.title('Repartition des survivants par sexe')
# plt.show()

# sns.countplot(x='Survived', hue='Pclass', data=data)
# plt.title('Repartition des survivants par sexe')
# plt.show()
