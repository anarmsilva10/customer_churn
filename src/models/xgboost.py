from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.data.load_data import load_data
from src.data.models import split_dataset, evaluate_model


def model_baseline(X_train, X_test, y_train, y_test):
    # Baseline Model

    # scale_por_weight - (# non-churn / # churn = 5174 / 1869 = ~2.77)

    model = XGBClassifier(scale_pos_weight = 2.77)
    model.fit(X_train, y_train)

    report = evaluate_model(model, X_test, y_test)

    return (model, report)


def model_tuning(X_train, X_test, y_train, y_test):

    # n_estimators - number of trees
    # learning_rate - step size shrinkage
    # max_depth - maximum tree depth (complexity)
    # subsample - fraction of data used per tree
    # colsample_bytree - fraction of features per tree
    # gamma - minimum loss reduction required for tree splits
    # reg_alpha - L1 regularisation
    # reg_lambda - L2 regularisation

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3, 1],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    }

    base_model = XGBClassifier(scale_pos_weight = 2.77)

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