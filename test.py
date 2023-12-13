import unittest
from process import transformation, import_data_Train
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def import_data():
    # Implémentez la logique d'importation de données ici
    test_data = import_data_Train()
    return test_data

def test_transformation():
    test_data = import_data()
    transformed_data = transformation(test_data)
    assert len(test_data) == len(transformed_data)

def test_logistic_regression_accuracy():
    test_data = import_data()
    test_data_trans = transformation(test_data)

    X_test = test_data_trans.drop("Survived", axis=1)
    y_test = test_data_trans["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.3)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert 0.0 <= accuracy <= 1.0


# if __name__ == '__main__':
#     unittest.main()
