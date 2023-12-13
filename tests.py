from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from explore import import_data_Train
from process import transformation


def test_drop_columns():
    # Load the training data
    data_train = import_data_Train()

    # Apply transformation to drop columns
    transformed_data = transformation(data_train)
    assert 'Name' not in transformed_data.columns

    # Add more assertions for the dropped columns


def test_sex_replacement():
    # Load the training data
    data_train = import_data_Train()

    # Apply transformation to replace 'Sex' values
    transformed_data = transformation(data_train)
    assert all(transformed_data['Sex'].isin([0, 1]))

    # Add more assertions for the 'Sex' replacement


def test_missing_values():
    # Load the training data
    data_train = import_data_Train()

    # Apply transformation to handle missing values
    transformed_data = transformation(data_train)
    assert transformed_data['Age'].isnull().sum() == 0

    # Add more assertions for missing values in 'Age'


def test_logistic_regression():
    # Load the training data
    data_train = import_data_Train()

    # Apply transformation
    data_train_trans = transformation(data_train)

    # Split data into training and testing sets
    X = data_train_trans.drop("Survived", axis=1)
    y = data_train_trans["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Instantiate Logistic Regression model
    log_reg = LogisticRegression()

    # Train the model
    log_reg.fit(X_train, y_train)

    # Predict using test data
    y_pred = log_reg.predict(X_test)

    # Test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.0 and accuracy <= 1.0

    # Test confusion matrix and classification report
    confusion_mat = confusion_matrix(y_test, y_pred)
    assert confusion_mat.shape == (2, 2)  # Assuming binary classification

    classification_rep = classification_report(y_test, y_pred)
    assert classification_rep != ""  # Ensure classification report is not empty

    # Test cross-validation
    scores = cross_val_score(log_reg, X, y, cv=5)
    for score in scores:
        assert score >= 0.0 and score <= 1.0
