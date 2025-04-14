import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# running this simple script to make sure webcam works and that everything is set up correctly.

print("OpenCV version:", cv2.__version__)
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Test webcam (if available)
try:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        print("Webcam is working")
    cap.release()
except:
    print("Webcam test failed")