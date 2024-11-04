import cv2
from ultralytics import YOLO
import sqlite3
import time

# Load your trained YOLO model
model = YOLO('runs/detect/train4/weights/best.pt')

# Define the objects for the front (3 items) and back (5 items)
front_objects = ['Ownermanual', 'Ownermanual','FirstaidKit']
back_objects = ['jacky', 'Towhook', 'Wheelspanner','Warningtriangle','FullKit']

# Function to check for missing objects
def check_missing(detections, category):
    detected_objects = [det.label for det in detections]
    missing_objects = [obj for obj in category if obj not in detected_objects]
    return missing_objects

# Function to store images in the database
def store_images_in_db(front_image, back_image, serial_number):
    conn = sqlite3.connect('object_detection.db')
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serial_number TEXT,
            front_image BLOB,
            back_image BLOB,
            timestamp TEXT
        )
    ''')

    # Convert images to binary
    _, front_img_encoded = cv2.imencode('.jpg', front_image)
    front_img_blob = front_img_encoded.tobytes()

    _, back_img_encoded = cv2.imencode('.jpg', back_image)
    back_img_blob = back_img_encoded.tobytes()

    # Insert into table
    cursor.execute('''
        INSERT INTO results (serial_number, front_image, back_image, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (serial_number, front_img_blob, back_img_blob, time.strftime('%Y-%m-%d %H:%M:%S')))

    conn.commit()
    conn.close()

# Initialize front and back cameras
cap_front = cv2.VideoCapture(0)
cap_back = cv2.VideoCapture(0)

while cap_front.isOpened() and cap_back.isOpened():
    ret_front, frame_front = cap_front.read()
    ret_back, frame_back = cap_back.read()

    # Detect objects using YOLO on front and back images
    results_front = model(frame_front)
    missing_front = check_missing(results_front[0].boxes, front_objects)

    results_back = model(frame_back)
    missing_back = check_missing(results_back[0].boxes, back_objects)

    # If any object is missing, display the missing items
    if missing_front or missing_back:
        if missing_front:
            cv2.putText(frame_front, f"Missing (Front): {', '.join(missing_front)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if missing_back:
            cv2.putText(frame_back, f"Missing (Back): {', '.join(missing_back)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # If no objects are missing, save the images
        cv2.putText(frame_front, "All objects detected. Saving images...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_back, "All objects detected. Saving images...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        store_images_in_db(frame_front, frame_back, "1234567890")

    # Display the frames
    cv2.imshow("Front View YOLO", frame_front)
    cv2.imshow("Back View YOLO", frame_back)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_front.release()
cap_back.release()
cv2.destroyAllWindows()
