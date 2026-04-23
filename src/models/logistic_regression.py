from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from src.data.load_data import load_data
from src.data.models import split_dataset, evaluate_model

df = load_data('../data/processed/telco_clean.csv')

X_train, X_test, y_train, y_test = split_dataset(df)

def model_baseline(X_train, X_test, y_train, y_test):
    
    # Baseline Model
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)

    report = evaluate_model(model, X_test, y_test)

    return (model, report)

# Wide random search → identify promising region → narrow grid search

def model_tuning(X_train, X_test, y_train, y_test):

    # penalty - specifies the norm of the penalty ('elasticnet' adds both L1 and L2 penalty terms)
    # C - inverse of regularization strength
    # l1_ratio - Elastic-Net parameter (l1_ratio=1 gives a pure L1-penalty, l1_ratio=0 gives a pure L2-penalty. Any value between 0 and 1 gives an Elastic-Net penalty)
    # saga - algorithm to use in the optimization problem ('saga' is faster for large datasets)

    param_dist = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.1, 1, 10],
        'l1_ratio': [0, 0.5, 1],
        'solver': ['saga']
    }

    base_model = LogisticRegression(class_weight='balanced')

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