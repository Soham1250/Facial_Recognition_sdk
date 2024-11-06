import cv2
import os
import numpy as np  # type: ignore
import dlib  # type: ignore
import mysql.connector  # type: ignore
from scipy.spatial import distance  # type: ignore
import subprocess
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH, SAVE_PATH


# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def connect_db():
    conn = mysql.connector.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    return conn

# Check if files exist
def check_file(path):
    
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return False
    return True

if not (check_file(SHAPE_PREDICTOR_PATH) and check_file(FACE_REC_MODEL_PATH)):
    raise FileNotFoundError("One or more required model files are missing.")

# Load the models
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

# Extract face descriptors
def get_face_descriptor(img, detector, shape_predictor, face_rec_model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None, []  # No faces detected

    face_descriptors = []
    for face in faces:
        shape = shape_predictor(gray, face)  # shape_predictor returns a dlib.full_object_detection
        descriptor = face_rec_model.compute_face_descriptor(img, shape)  # Correctly passing dlib.full_object_detection
        face_descriptors.append(np.array(descriptor))
    
    return face_descriptors, faces

# Compute the distance between embeddings
def compute_distance(embedding1, embedding2):
    return distance.euclidean(embedding1, embedding2)

# Determine if embeddings match based on a threshold
def is_match(embedding1, embedding2, threshold=0.2):
    dist = compute_distance(embedding1, embedding2)
    return dist < threshold

# Fetch stored embeddings from the database
def fetch_embeddings_from_db(person_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT angle, embedding FROM face_embeddings WHERE person_id = %s
    """, (person_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# Search for the person in the database
def find_matching_person(new_embedding):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT person_id FROM face_embeddings")
    person_ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    for person_id in person_ids:
        embeddings = fetch_embeddings_from_db(person_id)
        for angle, stored_embedding in embeddings:
            if is_match(new_embedding, np.array(stored_embedding)):
                return person_id
    return None

def run_face_recognition_script():
    script_path = "src/Face_Recognition.py"  
    subprocess.run(["python", script_path])


# Main function
def main():
    # Load face recognition models
    detector, shape_predictor, face_rec_model = load_models()

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

        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            # Crop the face region
            face_image = capture_face(frame, (x, y, w, h))
            image_path = save_face_image(face_image)
            # cv2.imshow('Captured Face', face_image)
           
            run_face_recognition_script()

        # Show the live feed
        cv2.imshow('Live Feed', frame)

        # Break loop on keypress (e.g., 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
