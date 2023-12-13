import pandas as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from explore import import_data_Train
from process import transformation

#Load data file
data_train = import_data_Train()

data_train_trans = transformation(data_train)

X = data_train_trans.drop("Survived", axis=1) 
y = data_train_trans["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))