from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_data
from src.data.models import split_dataset, evaluate_model

df = load_data('../data/processed/telco_clean.csv')

X_train, X_test, y_train, y_test = split_dataset(df)

def model_baseline(X_train, X_test, y_train, y_test):
    
    # Baseline Model
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    report = evaluate_model(model, X_test, y_test)

    return (model, report)


def model_tuning(X_train, X_test, y_train, y_test):

    # solver - algorithm to use in the optimization problem ('saga' is faster for large datasets)
    # penalty - specifies the norm of the penalty ('elasticnet' adds both L1 and L2 penalty terms)
    # C - inverse of regularization strength
    # l1_ratio - Elastic-Net parameter (l1_ratio=1 gives a pure L1-penalty, l1_ratio=0 gives a pure L2-penalty. Any value between 0 and 1 gives an Elastic-Net penalty)

    param_dist = [
        #  lbfgs is the default solver and only supports L2 penalty
        {
            'solver': ['lbfgs'],
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10]
        },

        #  liblinear supports L1 and L2 penalty
        {
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10]
        },

        #  saga supports Elastic_Net, L1 and L2 penalty
        {
            'solver': ['saga'],
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10],
        },

        {
            'solver': ['saga'],
            'penalty': ['elasticnet'],
            'C': [0.01, 0.1, 1, 10],
            'l1_ratio': [0, 0.5, 1]
        }
    ]

    # Scaling Data to enhance the model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = LogisticRegression(class_weight='balanced', max_iter=1000)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1',
        cv=5,
        random_state=42
    )

    search.fit(X_train_scaled, y_train)

    best_model = search.best_estimator_
    print('Best parameters:', search.best_params_)

    report = evaluate_model(best_model, X_test_scaled, y_test)

    return (best_model, report)