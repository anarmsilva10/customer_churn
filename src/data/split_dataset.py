from sklearn.model_selection import train_test_split

def split_dataset(df):

    X = df.drop('Churn', axis=1)  
    y = df['Churn']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test