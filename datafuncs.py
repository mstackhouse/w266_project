import pandas as pd
import ast

def read_data(datafile):
    ''' Helper function to properly read in the CSV post processed data'''
    # Read
    df = pd.read_csv(datafile)
    # Convert strings to lists
    df['SYMPTOMS'] = df['SYMPTOMS'].apply(lambda x: ast.literal_eval(x))
    df['VERSIONS'] = df['VERSIONS'].apply(lambda x: ast.literal_eval(x))
    return df