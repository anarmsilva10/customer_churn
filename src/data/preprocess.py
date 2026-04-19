from sklearn.preprocessing import LabelEncoder
import pandas as pd


def encode_binary(df):

    """
    Encoding Binary Variables.

    Args:
        Dataframe (df): Telco customer dataset.

    Returns:
        Dataframe (df): Telco customer dataset with the encoding binary variables.
        binary_columns: Binary Variables in the dataset.

    """

    binary_columns = [
        col for col in df.select_dtypes(include='object').columns
        if df[col].nunique() == 2
    ]

    encode = LabelEncoder()

    for col in binary_columns:
        df[col] = encode.fit_transform(df[col])

    
    return df, binary_columns


def encode_categorical(df, drop_first=True):

    """
    One-Hot Encoding for Categorical Variables.

    Args:
        DataFrame (df): Telco customer dataset.
        drop_first (default: True): when true the original columns are drop.

    Returns:
        DataFrame (df): Telco customer dataset with the encoding categorical variables.
        categorical_columns: Categorical Variables in the dataset.
    """

    categorical_columns = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if df[col].nunique() > 2
    ]

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=drop_first)

    return df, categorical_columns