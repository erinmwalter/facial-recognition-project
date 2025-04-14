from flask import Blueprint, render_template, request, redirect, url_for
from src.model.model_prediction import load_model, predict_emotions
from src.utils.face_utils import detect_faces, preprocess_face
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    faces = detect_faces(image)
        
    # Create a copy for drawing on
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
            
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            processed_face = preprocess_face(face_roi) 
            emotion, confidence = predict_emotions(processed_face, model)
                
            # Draw rectangle and emotion text
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(display, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
                
            # Save face details
            results['faces'].append({
                'emotion': emotion,
                'confidence': confidence
            })

            face_descriptions.append(f"Face {idx} is {emotion.lower()}")

        results['summary'] = f"We found {len(faces)} face(s). " + " ".join(face_descriptions)

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
