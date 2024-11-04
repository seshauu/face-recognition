import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load the trained model (ensure the path is correct)
model = YOLO('runs/detect/train6/weights/best.pt')  # Replace with your trained model path

# Paths to the front and back videos
front_video_path = "videos/WhatsApp Video 2024-09-17 at 1.30.06 PM.mp4"  # Replace with your front video file path
back_video_path = "videos/back.mp4"  # Replace with your back video file path

# Set the desired window size (optional)
screen_width, screen_height = 1280, 720

# Dictionary of objects to track
front_objects = {'Ownermanual': 2, 'FirstaidKit': 1}
back_objects = {'Jacky': 1, 'Towhook': 1, 'Wheelspanner': 1, 'Warningtriangle': 1, 'FullKit': 1, 'Wheelcup': 4}

# Directory to save captured photos
save_dir = 'captured_photos'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Threshold for accuracy (confidence)
accuracy_threshold = 0.50

# Counter for specific objects (Ownermanual and Wheelcup)
object_counts = {
    'Ownermanual': 0,
    'Wheelcup': 0
}


# Function to process a video and save detected object photos
def process_video(video_path, objects_to_detect, photo_prefix):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    detected_objects = set()  # To track detected objects

    while cap.isOpened():
        # Read each frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if the video ends

        # Run object detection on the current frame
        results = model(frame)

        # Annotate the detections on the frame
        for result in results:
            annotated_frame = result.plot()

            # Resize the image to fit within the screen (adjust to your screen size)
            h, w, _ = annotated_frame.shape
            if w > screen_width or h > screen_height:
                scaling_factor = min(screen_width / w, screen_height / h)
                new_width = int(w * scaling_factor)
                new_height = int(h * scaling_factor)
                annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))

            # Check if the specified objects are detected with confidence > accuracy_threshold
            for label in objects_to_detect.keys():
                for box in result.boxes:
                    class_id = int(box.cls[0])  # Get class ID
                    class_name = result.names[class_id]  # Map class ID to name
                    confidence = box.conf[0]  # Get the confidence of detection

                    # If a specified object is detected with confidence > 75% and hasn't been saved yet
                    if class_name == label and confidence > accuracy_threshold and label not in detected_objects:
                        detected_objects.add(label)
                        photo_path = os.path.join(save_dir, f"{photo_prefix}_{label}.png")
                        cv2.imwrite(photo_path, frame)  # Save the frame as an image
                        print(f"{label} detected with {confidence:.2f} confidence. Photo saved to {photo_path}")

                        # Increment the count if it's Ownermanual or Wheelcup
                        if label == 'Ownermanual':
                            object_counts['Ownermanual'] += 1
                        elif label == 'Wheelcup':
                            object_counts['Wheelcup'] += 1

            # Show the frame with YOLO detections (optional)
            cv2.imshow(f"Video Detection - {photo_prefix}", annotated_frame)

        # Break the loop if 'q' is pressed or all objects are detected
        if cv2.waitKey(1) & 0xFF == ord('q') or len(detected_objects) == len(objects_to_detect):
            break

    # Release video capture and close window
    cap.release()
    cv2.destroyAllWindows()

    # If all objects are detected, combine the photos
    if len(detected_objects) == len(objects_to_detect):
        combine_photos(photo_prefix, detected_objects)

    # Show missing objects in terminal
    missing_objects = set(objects_to_detect.keys()) - detected_objects
    if missing_objects:
        print(f"Missing objects in {photo_prefix}: {', '.join(missing_objects)}")
    else:
        print(f"All objects detected in {photo_prefix}.")


# Function to combine photos into one image
def combine_photos(photo_prefix, detected_objects):
    images = []
    for label in detected_objects:
        photo_path = os.path.join(save_dir, f"{photo_prefix}_{label}.png")
        if os.path.exists(photo_path):
            img = cv2.imread(photo_path)
            if img is not None:
                images.append(img)

    if images:
        # Combine images horizontally or vertically (based on your requirement)
        combined_image = np.hstack(images)  # Combine horizontally (use np.vstack for vertical)
        combined_image_path = os.path.join(save_dir, f"{photo_prefix}_combined.png")
        cv2.imwrite(combined_image_path, combined_image)
        print(f"Combined photo saved to {combined_image_path}")


# Process the front video for front objects
process_video(front_video_path, front_objects, 'front')

# Process the back video for back objects
process_video(back_video_path, back_objects, 'back')

# Display the final counts for Ownermanual and Wheelcup
print(f"Final counts: Ownermanual: {object_counts['Ownermanual']}, Wheelcup: {object_counts['Wheelcup']}")
