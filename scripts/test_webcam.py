import cv2
import numpy as np
import argparse
from src.model.model_prediction import load_model
from src.utils.face_utils import preprocess_face
from src.data.dataset import categories

def test_webcam(model=None):
    """
    Test model with webcam feed
    
    Parameters:
    model: Trained model (loads default if None)
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test facial emotion recognition with webcam')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    args = parser.parse_args()
    
    if args.model:
        model = load_model(args.model)
    else:
        model = load_model()
    
    test_webcam(model)