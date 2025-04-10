import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split

def create_model(input_shape, num_classes):
    """
    Create a CNN model for facial expression recognition
    
    Parameters:
    input_shape: Shape of input images (height, width, channels)
    num_classes: Number of emotion classes
    
    Returns:
    model: Compiled keras model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Classification block
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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