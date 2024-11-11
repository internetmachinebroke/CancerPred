import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history):
    """
    Plots the training and validation metrics
    """
    metrics = ['loss', 'accuracy', 'auc']
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model and produces performance metrics
    """
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_binary))
    
    cm = confusion_matrix(y_val, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()