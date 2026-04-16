from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def split_dataset(df):

    X = df.drop('Churn', axis=1)  
    y = df['Churn']  

    # stratify helps address class imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        stratify=y,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, output_dir=True):

    prediction = model.predict(X_test)

    if output_dir ==  True:
        report = classification_report(y_test, prediction, output_dict=True)
    else:
        print(classification_report(y_test, prediction))

    cm = confusion_matrix(y_test, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(type(model).__name__)
    plt.show()

    return report