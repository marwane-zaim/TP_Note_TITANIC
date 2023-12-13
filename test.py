import unittest
import pandas as pd
from process import transformation, import_data_Train
import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import cross_val_score

class TestTransformations(unittest.TestCase):
    def setUp(self):
        # Setup code if needed
        pass

    def test_transformation(self):
        # Load test data
        test_data = import_data_Train()

        # Apply the transformation
        transformed_data = transformation(test_data)

        # Add your assertions here
        self.assertEqual(len(test_data), len(transformed_data))
        # Add more assertions based on your specific transformation logic

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Load test data
        test_data = import_data_Train()

        # Apply the transformation
        self.test_data_trans = transformation(test_data)

        # Set up training and testing sets
        X_test = self.test_data_trans.drop("Survived", axis=1)
        y_test = self.test_data_trans["Survived"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_test, y_test, test_size=0.3)

        # Instantiate the logistic regression model
        self.log_reg = LogisticRegression()

    def test_logistic_regression_accuracy(self):
        # Fit the model
        self.log_reg.fit(self.X_train, self.y_train)

        # Predict
        y_pred = self.log_reg.predict(self.X_test)

        # Check accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertTrue(accuracy >= 0.0 and accuracy <= 1.0)

    def test_logistic_regression_confusion_matrix(self):
        # Fit the model
        self.log_reg.fit(self.X_train, self.y_train)

        # Predict
        y_pred = self.log_reg.predict(self.X_test)

        # Check confusion matrix
        confusion_mat = confusion_matrix(self.y_test, y_pred)
        self.assertEqual(confusion_mat.shape, (2, 2))  # Assuming binary classification

    def test_logistic_regression_classification_report(self):
        # Fit the model
        self.log_reg.fit(self.X_train, self.y_train)

        # Predict
        y_pred = self.log_reg.predict(self.X_test)

        # Check classification report
        classification_rep = classification_report(self.y_test, y_pred)
        self.assertNotEqual(classification_rep, "")

    def test_logistic_regression_cross_val_score(self):
        # Cross-validate the model
        scores = cross_val_score(self.log_reg, self.X_test, self.y_test, cv=5)

        # Check if scores are within the expected range
        for score in scores:
            self.assertTrue(score >= 0.0 and score <= 1.0)

if __name__ == '__main__':
    unittest.main()
