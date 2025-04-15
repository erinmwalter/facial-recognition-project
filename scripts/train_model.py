import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.data.dataset import load_data, categories
from src.model.model_training import train_model
from src.model.model_prediction import evaluate_model, predict_emotions
from src.visualization.plotting import plot_training_history, plot_confusion_matrix, show_misclassified
from src.model.model_definition import create_model


def explore_data(output_dir):
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
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    plt.close()
    
    # Save class distribution
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip([categories[u] for u in unique], counts))
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.bar(distribution.keys(), distribution.values())
    plt.title('Emotion Distribution in Dataset')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution.png'))
    plt.close()
    
    # Save distribution data as JSON
    with open(os.path.join(output_dir, 'distribution.json'), 'w') as f:
        json.dump({k: int(v) for k, v in distribution.items()}, f)
    
    print("Class distribution:")
    for category, count in distribution.items():
        print(f"{category}: {count} images")
    
    return images, labels

def train_and_test():
    """Train and test the model"""
    # Create directories
    os.makedirs("models", exist_ok=True)
    output_dir = 'app/static/training_assets'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Load data
    images, labels = explore_data(output_dir)
    
    # Train the model
    print("Training model...")
    model, history = train_model(images, labels)
    
    # Save training history as JSON
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f)
    
    # Plot and save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    print("Training complete! Model saved to models/emotion_model.keras")
    
    # Test the model
    print("Testing model...")
    # Split into train and test sets
    _, test_images, _, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_images, test_labels)
    
    # Save evaluation metrics
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as f:
        json.dump({'accuracy': float(test_accuracy), 'loss': float(test_loss)}, f)
    
    # Make predictions
    pred_labels = model.predict(test_images)
    pred_classes = np.argmax(pred_labels, axis=1)
    
    # Generate classification report
    report = classification_report(test_labels, pred_classes, target_names=categories, output_dict=True)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f)
    
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_classes, target_names=categories))
    
    cm = confusion_matrix(test_labels, pred_classes)
    
    # Save confusion matrix data
    with open(os.path.join(output_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump({'matrix': cm.tolist(), 'labels': categories}, f)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Find misclassified examples
    misclassified = np.where(test_labels != pred_classes)[0]
    misclassified_data = []
    
    # Save up to 20 misclassified images
    for i, idx in enumerate(misclassified[:20]):
        # Save the image
        img = test_images[idx].reshape(48, 48)
        img = (img * 255).astype(np.uint8)
        img_path = f'images/misclassified_{i}.png'
        plt.imsave(os.path.join(output_dir, img_path), img, cmap='gray')
        
        # Add to data
        misclassified_data.append({
            'image': img_path,
            'true_label': categories[test_labels[idx]],
            'pred_label': categories[pred_classes[idx]]
        })
    
    # Save misclassified data
    with open(os.path.join(output_dir, 'misclassified.json'), 'w') as f:
        json.dump(misclassified_data, f)
    
    # Plot grid of misclassified examples
    if len(misclassified) > 0:
        rows = min(4, (len(misclassified) + 3) // 4)
        cols = min(4, len(misclassified))
        
        plt.figure(figsize=(12, 3*rows))
        
        for i, idx in enumerate(misclassified[:16]):
            plt.subplot(rows, cols, i+1)
            img = test_images[idx].reshape(48, 48)
            plt.imshow(img, cmap='gray')
            plt.title(f"True: {categories[test_labels[idx]]}\nPred: {categories[pred_classes[idx]]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'misclassified_grid.png'))
        plt.close()
    
    print(f"All visualizations and data saved to {output_dir}")
    return model

if __name__ == "__main__":
    train_and_test()