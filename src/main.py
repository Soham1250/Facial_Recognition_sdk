import cv2
import os
import numpy as np  # type: ignore
import dlib  # type: ignore
import subprocess
from Face_Recognition import recognize_face  # Import recognize_face from face_recognition.py
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH, SAVE_PATH

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face recognition models
def load_models():
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
    return detector, shape_predictor, face_rec_model

# Detect faces in a frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def save_face_image(face_image):
    face_image_path = os.path.join(SAVE_PATH, "temp.jpg")
    try:
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        cv2.imwrite(face_image_path, face_image)
        return face_image_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def capture_face(frame, face_coords):
    (x, y, w, h) = face_coords
    face_image = frame[y:y + h, x:x + w]
    return face_image

# Main function
def main():
    # Load face recognition models
    detector, shape_predictor, face_rec_model = load_models()

    # Initialize counters for accuracy calculation
    total_scans = [0]  # Mutable list for passing by reference
    successful_scans = [0]

    # Open the video stream
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces in the frame
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            # Capture and save the face image
            face_image = capture_face(frame, (x, y, w, h))
            image_path = save_face_image(face_image)

            # Run face recognition on the saved face image and update counters
            recognize_face(image_path, total_scans, successful_scans)

            # Draw rectangle around face for visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the live feed with rectangles around detected faces
        cv2.imshow('Live Feed', frame)

        # Break loop on keypress (e.g., 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close the display window
    cap.release()
    cv2.destroyAllWindows()

    # Calculate and display accuracy after stopping the recognition
    if total_scans[0] > 0:
        accuracy = (successful_scans[0] / total_scans[0]) * 100
    else:
        accuracy = 0.0

    print(f"Total Scans: {total_scans[0]}")
    print(f"Successful Scans: {successful_scans[0]}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
