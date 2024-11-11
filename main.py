import tensorflow as tf
import numpy as np
import os
from data_loader import load_and_augment_data
from data_processor import preprocess_data
from model import create_cnn_model
from trainer import train_model
from visualizer import plot_training_history, evaluate_model
from predictor import predict_single_image

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    print("Loading data...")
    X, y = load_and_augment_data('GT_path')
    
    # Preprocess and split data
    print("Preprocessing data...")
    X_train, X_val, y_train, y_val = preprocess_data(X, y)
    
    # Create and compile model
    print("Creating model...")
    model = create_cnn_model()
    model.summary()
    
    # Train model
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_val, y_val)
    
    print("\nTraining complete! The best model has been saved as 'best_model.keras'")

if __name__ == "__main__":
    main()