import cv2
import numpy as np
import face_recognition
import os

print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

# Image path
image_path = 'images/Elon Musk.jpg'

# Try to read the image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image from path: {image_path}.")
else:
    print(f"Image loaded with shape: {image.shape}")

    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Image converted to RGB format.")

    # Detect faces
    try:
        face_locations = face_recognition.face_locations(rgb_image)
        print(f"Number of faces detected: {len(face_locations)}")
    except Exception as e:
        print(f"Error during face detection: {e}")
