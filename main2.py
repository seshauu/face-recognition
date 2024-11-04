import cv2
import pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from simple_facerec import SimpleFacerec
import os
import torch

# Check device
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print(f"Using device: {device} - {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")  # Use CPU
    print(f"Using device: {device}")

# Initialize the SimpleFacerec class and load encoding images from the "images" folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Email Setup
def send_email(subject, body, recipient_emails, image_path):
    sender_email = "praveentansam@gmail.com"  # Replace with your email
    sender_password = "dkoy yloq vhsl dhrf"  # Replace with your app password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)  # Join multiple recipients with commas
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # Open the image file and attach it to the email
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
            msg.attach(img)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)  # Use your email provider's SMTP server and port
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_emails, text)
        server.quit()
        print("Email sent successfully to all recipients!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load the camera feed (change the index based on your camera setup)
cap = cv2.VideoCapture(0)

# Keep track of whether we have already announced an unknown person
last_detection = None
unknown_timer = 4  # Time to limit repeated announcements
email_timer = 1  # Timer to limit how often we send emails
image_count = 0  # To ensure unique image filenames

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect known faces in the frame using the CNN model for higher accuracy
    face_locations, face_names = sfr.detect_known_faces(frame, model="cnn", tolerance=0.4, unknown_threshold=0.6)

    # Display results and give voice feedback if an unknown person is detected
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Draw a rectangle around the face
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Check if the detected person is unknown and prevent repeated voice prompts
        if name == "Unknown":
            if last_detection != "Unknown" or unknown_timer >= 50:
                speak("Unauthorized person entry in the storage area")
                print("Unknown person detected. Sending email...")
                last_detection = "Unknown"
                unknown_timer = 0  # Reset timer after speaking

                # Save the current frame as an image
                image_path = f"unknown_person_{image_count}.jpg"
                cv2.imwrite(image_path, frame)
                image_count += 1

                # Send email when an unknown person is detected
                if email_timer >= 5:  # Adjust this value for faster email notifications
                    subject = "Unauthorized person entry in the storage area"
                    body = "An unknown person was detected by the system. See the attached image."
                    recipient_emails = ["praveen1996.pg@gmail.com", "seshanjai@gmail.com","Shanmugam.Murugan@rnaipl.com"] #Shanmugam.Murugan@rnaipl.com Add multiple recipients here
                    send_email(subject, body, recipient_emails, image_path)
                    email_timer = 0  # Reset the email timer after sending
                else:
                    print(f"Email not sent. Email timer is {email_timer}")

        else:
            last_detection = name  # Remember the last known person
            unknown_timer = 0  # Reset the timer if a known person is detected

    # Increment the unknown and email timers for delayed actions
    unknown_timer += 1
    email_timer += 1

    # Display the frame with rectangles
    cv2.imshow("Frame", frame)

    # Exit on pressing 'ESC'
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
