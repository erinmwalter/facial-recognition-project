import cv2
import numpy as np

def preprocess_face(face, target_size=(48, 48)):
    """
    Preprocess a face image for emotion prediction
    
    Parameters:
    face: BGR face image
    target_size: Target size for the model input (height, width)
    
    Returns:
    processed: Preprocessed face tensor (1, height, width, 1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize
    resized = cv2.resize(gray, target_size)
    
    # Normalize
    normalized = resized / 255.0
    
    # Reshape for model input
    processed = normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return processed

def detect_faces(image):
    # Load face cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,    
        minNeighbors=3,      
        minSize=(30, 30),  
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    print(f"[DEBUG] Detected {len(faces)} face(s)")
    return faces