import tensorflow as tf
import numpy as np
from src.data.dataset import categories

def load_model(model_path="models/emotion_model.keras"):
    """
    Load a trained model
    
    Parameters:
    model_path: Path to the saved model
    
    Returns:
    model: Loaded model
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model

def evaluate_model(model, test_images, test_labels):
    """
    Evaluate model performance on test data
    
    Parameters:
    model: Trained model
    test_images: Test image data
    test_labels: Test labels
    
    Returns:
    test_loss, test_accuracy: Evaluation metrics
    """
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    return test_loss, test_accuracy

def predict_emotions(model, images):
    """
    Predict emotions for a batch of images
    
    Parameters:
    model: Trained model
    images: Image data
    
    Returns:
    predictions: Predicted emotion indices
    """
    predictions = model.predict(images)
    return np.argmax(predictions, axis=1)