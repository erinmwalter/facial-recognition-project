import cv2
import tensorflow as tf
import numpy as np
import os

print("Testing project setup...")
print("OpenCV version:", cv2.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Test imports from reorganized structure
try:
    from src.data.dataset import categories
    print("Successfully imported categories:", categories)
    
    from src.utils.face_utils import preprocess_face
    print("Successfully imported preprocess_face")
    
    from src.model.model_definition import create_model
    print("Successfully imported create_model")
    
    # Check if model exists
    if os.path.exists("models/emotion_model.keras"):
        from src.model.model_prediction import load_model
        model = load_model()
        print("Successfully loaded model")
    else:
        print("Model file not found (this is OK if you haven't trained yet)")
    
    print("\nAll imports successful! Your project structure is working.")
except Exception as e:
    print(f"Error during import test: {e}")
    print("Please check your project structure and imports.")