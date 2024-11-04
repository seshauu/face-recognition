import cv2
from simple_facerec import SimpleFacerec

# Initialize the SimpleFacerec class and load encoding images from the "images" folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load the camera feed (change the index based on your camera setup)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect known faces in the frame
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Display results
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Draw a rectangle around the face
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Display the frame with rectangles
    cv2.imshow("Frame", frame)

    # Exit on pressing 'ESC'
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
