from flask import Blueprint, render_template, request, redirect, url_for, jsonify, Response
from src.model.model_prediction import load_model, predict_emotions
from src.utils.face_utils import detect_faces, preprocess_face
from src.data.dataset import categories
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import io
import threading
import time

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

camera = None
output_frame = None
lock = threading.Lock()
is_streaming = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('main.index'))

    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('main.index'))
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename) 
    file.save(filepath)

    image = cv2.imread(filepath)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
            image, 
            scaleFactor=1.1, 
            minNeighbors=8,
            minSize=(60, 60)
        )
        
    # Create a copy for drawing on in case we need original
    display = image.copy()
        
    # Results to pass to template
    results = {
            'filename': filename,
            'face_count': len(faces),
            'faces': []
        }
    
    result_filename = f'result_{filename}'
    result_filepath = os.path.join(UPLOAD_FOLDER, result_filename)
    
    if len(faces) > 0:
        model = load_model()
        face_descriptions = []
            
        for i, face in enumerate(faces):
            # Extract face region
            x, y, w, h = face
            face_roi = image[y:y+h, x:x+w]
            processed_face = preprocess_face(face_roi) 
            
            # Predict the emotion and then also get confidence of prediction
            prediction = model.predict(processed_face)[0]
            emotion_idx = np.argmax(prediction)
            emotion = categories[emotion_idx]
            confidence = prediction[emotion_idx] * 100

            # Draw rectangle and emotion text (uses confidence and emotion above)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(display, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        5, (0, 255, 0), 11)
                
            # Save face details
            results['faces'].append({
                    'emotion': emotion,
                    'confidence': confidence
            })

            face_descriptions.append(f"Face {i+1} is {emotion.lower()}")

        results['summary'] = f"{len(faces)} face(s) found. " + " ".join(face_descriptions)

        # Save the processed image
        result_filename = 'result_' + filename
        result_filepath = os.path.join(UPLOAD_FOLDER, result_filename)
        
        cv2.imwrite(result_filepath, display)
        
        # Pass results to template
        return render_template('index.html', 
                               uploaded_image=url_for('static', filename=f'uploads/{filename}'),
                               result_image=url_for('static', filename=f'uploads/{result_filename}'),
                               results=results)
    
    results['summary'] = "Here is your uploaded image. Error: no faces detected."

    return render_template('index.html',
                           uploaded_image=url_for('static', filename=f'uploads/{filename}'),
                           results=results)

@main.route('/start_stream', methods=['POST'])
def start_stream():
    global camera, is_streaming
    
    if camera is not None and camera.isOpened():
        return jsonify({'status': 'already_running'})
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return jsonify({'status': 'error', 'message': 'Could not open webcam'})
    
    # Start the webcam processing thread
    is_streaming = True
    threading.Thread(target=process_webcam, daemon=True).start()
    
    return jsonify({'status': 'success'})

@main.route('/stop_stream', methods=['POST'])
def stop_stream():
    global camera, is_streaming
    
    # Stop streaming
    is_streaming = False
    
    # Release camera if it exists
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({'status': 'success'})

@main.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def process_webcam():
    """Process webcam frames and perform emotion detection"""
    global camera, output_frame, lock, is_streaming
    
    # Load model
    model = load_model()
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while is_streaming:
        # Read frame
        success, frame = camera.read()
        if not success:
            break
        
        # Create a copy for drawing on
        display = frame.copy()
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            frame, 
            scaleFactor=1.1, 
            minNeighbors=6,
            minSize=(60, 60)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess face
                processed_face = preprocess_face(face_roi)
                
                # Predict emotion
                prediction = model.predict(processed_face)[0]
                emotion_idx = np.argmax(prediction)
                emotion = categories[emotion_idx]
                confidence = prediction[emotion_idx] * 100
                
                if confidence > 60.0:
                    # Draw rectangle and emotion text
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"{emotion} ({confidence:.1f}%)"
                    cv2.putText(display, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Update the output frame with thread safety
        with lock:
            output_frame = display.copy()
        
        # Sleep to reduce CPU usage
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    """Generate frames for video streaming"""
    global output_frame, lock
    
    while True:
        # Wait until we have a processed frame
        if output_frame is None:
            time.sleep(0.1)
            continue
        
        # Encode the frame as JPEG
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
        if not flag:
            continue
            
        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encoded_image) + b'\r\n')
        
        # Sleep to control frame rate
        time.sleep(0.03)  # ~30 FPS