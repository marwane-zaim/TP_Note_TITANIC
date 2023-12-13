import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import cross_val_score
from explore import import_data_Test, import_data_Train
from process import transformation

def test_drop_columns():
    data_train = import_data_Train()
    
    transformed_data = transformation(data_train)
    assert 'Name' not in transformed_data.columns

    # Ajoutez d'autres assertions pour les colonnes supprimées

def test_sex_replacement():
    data_train = import_data_Train()

    transformed_data = transformation(data_train)
    assert all(transformed_data['Sex'].isin([0, 1]))

    # Ajoutez d'autres assertions pour le remplacement de 'Sex'

def test_missing_values():
    data_train = import_data_Train()

    transformed_data = transformation(data_train)
    assert transformed_data['Age'].isnull().sum() == 0

    # Ajoutez d'autres assertions pour les valeurs manquantes dans 'Age'

def test_logistic_regression():
    # Chargez les données d'entraînement
    data_train = import_data_Train()

    # Appliquez la transformation
    data_train_trans = transformation(data_train)

    # Séparez les données en ensembles d'entraînement et de test
    X = data_train_trans.drop("Survived", axis=1) 
    y = data_train_trans["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Instanciez le modèle de régression logistique
    log_reg = LogisticRegression()

    # Entraînez le modèle
    log_reg.fit(X_train, y_train)

    # Prédisez avec les données de test
    y_pred = log_reg.predict(X_test)

    # Testez l'exactitude (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.0 and accuracy <= 1.0

    # Testez la matrice de confusion et le rapport de classification
    confusion_mat = confusion_matrix(y_test, y_pred)
    assert confusion_mat.shape == (2, 2)  # Assuming binary classification

    classification_rep = classification_report(y_test, y_pred)
    assert classification_rep != ""  # Vérifiez que le rapport de classification n'est pas vide

    # Testez la cross-validation
    scores = cross_val_score(log_reg, X, y, cv=5)
    for score in scores:
        assert score >= 0.0 and score <= 1.0