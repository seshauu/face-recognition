import cv2
from ultralytics import YOLO

# Load the trained model (ensure the path is correct)
model = YOLO('runs/detect/train6/weights/best.pt')  # Replace this with your trained model path

# Path to the video to run inference on
video_path = "videos/WhatsApp Video 2024-09-17 at 1.30.06 PM (online-video-cutter.com).mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set the desired window size (optional)
screen_width, screen_height = 1280, 720

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

        # Show the frame with YOLO detections
        cv2.imshow("Video Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
