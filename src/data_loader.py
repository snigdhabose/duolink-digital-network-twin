# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath, test_size=0.2):
    data = pd.read_csv(filepath, header=None)
    
    # The labels are in the second-to-last column
    labels = data.iloc[:, -2].astype(str)
    
    # Remove the last two columns from data (labels and difficulty level)
    data = data.iloc[:, :-2]
    
    # Automatically detect and label-encode non-numeric columns in features
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    
    # Clean labels: remove periods, convert to lowercase, strip whitespace
    labels = labels.str.strip().str.lower().str.rstrip('.')
    
    # Map labels to binary classes: 0 for 'normal', 1 for anomalies
    y = labels.apply(lambda x: 0 if x == 'normal' else 1)
    
    # Debug: Print unique labels after cleaning and mapping
    print("Unique labels after cleaning:", labels.unique())
    print("Unique labels after mapping:", np.unique(y))
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    
    # Split the data into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Debug: Print counts of each class to verify correct label mapping
    print("Counts in y_train:", np.bincount(y_train))
    print("Counts in y_test:", np.bincount(y_test))
    
    return X_train, X_test, y_train, y_test
