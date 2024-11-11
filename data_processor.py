from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_data(X, y):
    """
    Preprocesses the image data and splits it into training and validation sets
    """
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val