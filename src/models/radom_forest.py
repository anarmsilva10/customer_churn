from src.data.load_data import load_data
from src.data.split_dataset import split_dataset

df = load_data('data/processed/telco_clean.csv')

X_train, X_test, y_train, y_test = split_dataset(df)
