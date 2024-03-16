import face_recognition
import os
import cv2
from datetime import datetime

# Function to load images from a directory
def load_images_from_folder(folder_path):
    images = []
    encodings = []
    names = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            names.append(os.path.splitext(filename)[0])  # Extract name from filename (without extension)
            images.append(image)
            encodings.append(encoding)

    return images, encodings, names

# Function to mark attendance with     timestamp
def mark_attendance(name):
    with open('attendance.csv', 'a') as file:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{name},{now}\n")

# Set up known images and encodings
known_images_folder = 'static/faces'  # Replace with your folder path
known_images, known_encodings, known_names = load_images_from_folder(known_images_folder)

# Initialize video capture and face recognition process
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Convert frame to RGB format for face recognition
    rgb_frame = frame[:, :, ::-1]

    # Find all faces and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Match the face encoding with the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, get the name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

            # Mark attendance for the recognized person
            mark_attendance(name)

        # Draw rectangle around the face and display name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display name below the face rectangle
        font_scale = 1.0
        font_thickness = 2
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness)

    # Display the resulting frame
    cv2.imshow('Attendance System', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
