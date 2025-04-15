import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from src.model.model_definition import create_model

def train_model(images, labels, model_save_path="models/emotion_model.keras"):
    """
    Train the facial expression recognition model
    
    Parameters:
    images: Preprocessed image data
    labels: Image labels
    model_save_path: Path to save the trained model
    
    Returns:
    model: Trained model
    history: Training history
    """
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(labels))
    
    # Create the model
    model = create_model(input_shape, num_classes)
    print(model.summary())
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    return model, history