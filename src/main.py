import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
from data_loader import load_data, categories
from training import train_model
from testing import (load_model, evaluate_model, predict_emotions, 
                    plot_confusion_matrix, show_misclassified, test_webcam, preprocess_face)
from sklearn.model_selection import train_test_split
from tkinter import Tk, filedialog

def plot_training_history(history):
    """Plot training and validation accuracy and loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

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

def analyze_image(file_path=None):
    """Analyze a single image for facial expressions
    
    Parameters:
    file_path: Optional path to image file. If None, user will be prompted.
    """
    # Load model
    model = load_model()
    
    # If no file path provided, either ask for input or show file dialog
    if file_path is None:
        # First try command line input
        file_path = input("Enter the path to an image file (or press Enter to open file dialog): ").strip()
        
        # If still empty, use file dialog
        if not file_path:
            # Create Tkinter root window (will be hidden)
            root = Tk()
            root.withdraw()
            
            # Open file dialog for image selection
            file_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
            )
    
    if not file_path:
        print("No image selected. Exiting...")
        return
    
    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not load image from {file_path}")
        return
    
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Create a copy for drawing on
    display = image.copy()
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        # Show the image anyway
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        plt.title("No faces detected")
        plt.axis('off')
        plt.show()
        return
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess face
        processed_face = preprocess_face(face_roi)
        
        # Predict emotion
        prediction = model.predict(processed_face)[0]
        emotion_idx = np.argmax(prediction)
        emotion = categories[emotion_idx]
        confidence = prediction[emotion_idx] * 100
        
        # Draw rectangle and emotion text
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(display, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    # Display results
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.title("Facial Emotion Analysis")
    plt.axis('off')
    plt.show()
    
    print(f"Analysis complete! Found {len(faces)} face(s).")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Facial Expression Recognition')
    parser.add_argument('--mode', type=str, default='train_test',
                      help='Mode: train_test, analyze_image, or webcam')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to image file for analysis (only used with --mode analyze_image)')
    args = parser.parse_args()
    
    # Mode selection
    mode = args.mode.lower()
    
    if mode == 'train_test':
        train_and_test()
    elif mode == 'analyze_image':
        analyze_image(args.image)
    elif mode == 'webcam':
        model = load_model()
        test_webcam(model)
    else:
        print(f"Invalid mode: {mode}")
        print("Available modes: train_test, analyze_image, webcam")

if __name__ == "__main__":
    main()