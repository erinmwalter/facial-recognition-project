import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from src.data.dataset import categories


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix
    
    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def show_misclassified(images, true_labels, pred_labels, n=16):
    """
    Show misclassified images
    
    Parameters:
    images: Image data
    true_labels: True labels
    pred_labels: Predicted labels
    n: Number of images to show
    """
    # Find misclassified examples
    misclassified = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified) == 0:
        print("No misclassified images found!")
        return
    
    # Select random misclassified examples
    indices = np.random.choice(misclassified, min(n, len(misclassified)), replace=False)
    
    # Plot
    rows = int(np.ceil(len(indices) / 4))
    plt.figure(figsize=(12, 3*rows))
    
    for i, idx in enumerate(indices):
        plt.subplot(rows, 4, i+1)
        img = images[idx].reshape(48, 48)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {categories[true_labels[idx]]}\nPred: {categories[pred_labels[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_webcam(model):
    """
    Test model with webcam feed
    
    Parameters:
    model: Trained model
    """
    # Load face cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for drawing on
        display = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # For each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Draw rectangle around face
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            try:
                # Preprocess face
                processed_face = preprocess_face(face_roi)
                
                # Predict emotion
                prediction = model.predict(processed_face)[0]
                emotion_idx = np.argmax(prediction)
                emotion = categories[emotion_idx]
                confidence = prediction[emotion_idx] * 100
                
                # Display emotion text
                text = f"{emotion} ({confidence:.1f}%)"
                cv2.putText(display, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Display the result
        cv2.imshow('Facial Emotion Recognition', display)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def preprocess_face(face, target_size=(48, 48)):
    """Preprocess a face image for prediction"""
    # Convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, target_size)
    
    # Normalize
    normalized = resized / 255.0
    
    # Reshape for model input
    processed = normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return processed

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
