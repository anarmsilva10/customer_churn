from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from src.data.load_data import load_data
from src.data.models import split_dataset, evaluate_model

df = load_data('../data/processed/telco_clean.csv')

X_train, X_test, y_train, y_test = split_dataset(df)

def model_baseline(X_train, X_test, y_train, y_test):

    model = svm.SVC(kernel='linear', class_weight='balanced')
    model.fit(X_train, y_train)

    report = evaluate_model(model, X_test, y_test)

    return (model, report)


def model_tuning(X_train, X_test, y_train, y_test):

    # Degree - controls the complexity of the boundary
    # C - "error budget" - creates a budget for the model to make mistakes
    # Gamma - decides how closely the model fits the training data
    # Kernel - creates different boundaries

    param_dist = {
        'degree': [0, 1, 6],
        'C': [0.1, 1, 10],
        'gamma': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
    }

    base_model = svm.SVC(class_weight='balanced')

    search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist, 
        n_iter=20,
        scoring='f1',
        cv=5,
        random_state=42
    )
    
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print('Best parameters:', search.best_params_)

    report = evaluate_model(best_model, X_test, y_test)

    return (best_model, report)