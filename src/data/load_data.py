import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path):
    
    """
    Load dataset.
    
    Arguments:
        - path: Path for the dataset file.
        
    Returns:
        - Dataframe (df) with the dataset.    
    """

    df = pd.read_csv(path)

    return df

def plot_feature(df, column):
    
    """
    Creates two diferent plot types depending on the variable type.
    
    Arguments:
        - DataFrame (df): Dataframe qith the dataset.
        - column: Column name.
        - target: Churn column.
        
    Returns:
        - Histplot for Numeric Variables
        - Countplot for Categorical Variables.    
    """
    
    if df[column].dtype in ['int64', 'float64']:
        
        plt.figure(figsize=(6,4))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribuition of {column}')
        plt.show()
        
    else:
        
        plt.figure(figsize=(6,4))
        sns.countplot(x=column, data=df)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
        plt.show()
        
        
def plot_vs(df, column, target='Churn'):
    
    """
    Creates two diferent plot types depending on the variable type.
    
    Arguments:
        - DataFrame (df): Dataframe qith the dataset.
        - column: Column name.
        - target: Churn column.
        
    Returns:
        - Histplot for Numeric Variables
        - Countplot for Categorical Variables.    
    """
    
    if df[column].dtype in ['int64', 'float64']:
        
        plt.figure(figsize=(6,4))
        sns.boxenplot(x=target, y=column, data=df)
        plt.title(f'Boxplot of {column} by {target}')
        plt.show()
        
    else:
        
        plt.figure(figsize=(6,4))
        sns.countplot(x=column, hue=target, data=df)
        plt.title(f'Distribution of {column} by {target}')
        plt.xticks(rotation=45)
        plt.show()