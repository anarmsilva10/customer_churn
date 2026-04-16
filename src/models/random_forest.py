from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.data.load_data import load_data
from src.data.models import split_dataset, evaluate_model

df = load_data('../data/processed/telco_clean.csv')

X_train, X_test, y_train, y_test = split_dataset(df)

def model_baseline(X_train, X_test, y_train, y_test):
    # Baseline Model
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    report = evaluate_model(model, X_test, y_test)

    return (model, report)

# Wide random search → identify promising region → narrow grid search

def model_tuning(X_train, X_test, y_train, y_test):

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    base_model = RandomForestClassifier(class_weight='balanced')

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1',
        cv=5,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print('Best parameters:', search.best_params_)

    report = evaluate_model(best_model, X_test, y_test)

    return (best_model, report)