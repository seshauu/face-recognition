import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize factor for faster processing

    def load_encoding_images(self, images_path):
        """
        Load encoding images from the specified path.
        Each person's images should be in a separate folder, and the folder name will be used as the person's name.
        :param images_path: Path to the folder containing person folders
        """
        # List all folders in the images directory (each folder is a person)
        people_folders = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
        print(f"Found {len(people_folders)} folders of people.")

        for folder_name in people_folders:
            folder_path = os.path.join(images_path, folder_name)
            # Load all images for this person
            image_files = glob.glob(os.path.join(folder_path, "*.*"))

            print(f"Loading {len(image_files)} images for {folder_name}...")

            for img_path in image_files:
                img = cv2.imread(img_path)

                # Check if the image was loaded correctly
                if img is None:
                    print(f"Could not load image {img_path}, skipping.")
                    continue

                # Convert the image to RGB format
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Try to get face encodings
                encodings = face_recognition.face_encodings(rgb_img)
                if encodings:
                    img_encoding = encodings[0]
                    self.known_face_encodings.append(img_encoding)
                    # Store the person's name (folder name)
                    self.known_face_names.append(folder_name)
                else:
                    print(f"No face found in {img_path}, skipping.")

        print("Encoding images loaded successfully.")

    def detect_known_faces(self, frame, model="hog", tolerance=0.4, unknown_threshold=0.6):
        """
        Detect faces in a frame and match them with known encodings.
        :param frame: Video frame to process
        :param model: Model to use for face detection ('hog' or 'cnn')
        :param tolerance: Tolerance for face matching (lower is stricter)
        :param unknown_threshold: Distance above which the face is labeled as "Unknown"
        :return: Tuple of (face_locations, face_names)
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and their encodings using the specified model ('hog' or 'cnn')
        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Use tolerance in compare_faces function
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"

            # Use the face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # If the best match is below the unknown threshold, assign the known name
            if matches[best_match_index] and face_distances[best_match_index] < unknown_threshold:
                name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"

            face_names.append(name)

        # Adjust face locations based on frame resizing
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        return face_locations.astype(int), face_names