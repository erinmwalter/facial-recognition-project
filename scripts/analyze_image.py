import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tkinter import Tk, filedialog

from src.data.dataset import categories
from src.model.model_prediction import load_model
from src.utils.face_utils import preprocess_face, detect_faces


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
    faces = detect_faces(image)
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze image for facial emotions')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    args = parser.parse_args()
    analyze_image(args.image)