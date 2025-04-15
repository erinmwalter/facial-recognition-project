import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# all categories and the file path for each of the jpg files
# each category is in a sub-folder. for instance, all "angry" pics are in angry folder
# dataset from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013/data
# it is the FER-2013 dataset
categories = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
file_path = "src/data/test"

def load_data(img_size=48):
    """
    Load images from dataset folders, process them

    Returns the images loaded and the labels
    """

    data = []
    labels = []

    # Loop through each emotion category
    for category_idx, category in enumerate(categories):  
        # Path to this category's folder
        path = os.path.join(file_path, category)
        print(f"Loading {category} images from {path}...")
        
        # Loop through all files in the folder
        for img_file in os.listdir(path):
            try:
                # Full path to image
                img_path = os.path.join(path, img_file)
                
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize image
                img = cv2.resize(img, (img_size, img_size))
                
                # Add to data list
                data.append(img)
                
                # Add corresponding label
                labels.append(category_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert lists to numpy arrays
    images = np.array(data)
    labels = np.array(labels)
    
    # Normalize pixel values to [0,1]
    images = images / 255.0
    
    # Reshape for CNN input (add channel dimension)
    images = images.reshape(images.shape[0], img_size, img_size, 1)
    
    print(f"Loaded {len(data)} images")
    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return images, labels