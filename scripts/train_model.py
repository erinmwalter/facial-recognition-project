import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data.dataset import load_data, categories
from src.model.model_training import train_model
from src.model.model_prediction import evaluate_model, predict_emotions
from src.visualization.plotting import plot_training_history, plot_confusion_matrix, show_misclassified


def explore_data():
    """Explore and visualize the dataset"""
    # Load the data
    print("Loading dataset...")
    images, labels = load_data()
    
    # Show sample images
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = images[i].reshape(48, 48)  # Remove channel dimension for display
        plt.imshow(img, cmap='gray')
        plt.title(categories[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip([categories[u] for u in unique], counts))
    print("Class distribution:")
    for category, count in distribution.items():
        print(f"{category}: {count} images")
    
    return images, labels

def train_and_test():
    """Train and test the model"""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load data
    images, labels = explore_data()
    
    # Train the model
    print("Training model...")
    model, history = train_model(images, labels)
    
    # Plot training history
    plot_training_history(history)
    
    print("Training complete! Model saved to models/emotion_model.keras")
    
    # Test the model
    print("Testing model...")
    # Split into train and test sets
    _, test_images, _, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Evaluate model
    evaluate_model(model, test_images, test_labels)
    
    # Make predictions
    pred_labels = predict_emotions(model, test_images)
    
    # Show classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_labels, target_names=categories))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, pred_labels)
    
    # Show misclassified examples
    show_misclassified(test_images, test_labels, pred_labels)

if __name__ == "__main__":
    train_and_test()