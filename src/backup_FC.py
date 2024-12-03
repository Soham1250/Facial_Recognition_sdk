import json
import cv2
import dlib  # type: ignore
import numpy as np  # type: ignore
import os
import mysql.connector  # type: ignore
from scipy.spatial import distance  # type: ignore
import subprocess
import time
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH, SAVE_PATH


def connect_db():
    conn = mysql.connector.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    return conn

# File paths
SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"

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

# Extract face descriptors
def get_face_descriptor(img, detector, shape_predictor, face_rec_model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None, []  # No faces detected

    face_descriptors = []
    for face in faces:
        shape = shape_predictor(gray, face)
        descriptor = face_rec_model.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(descriptor))
    
    return face_descriptors, faces

# Compute the distance between embeddings
def compute_distance(embedding1, embedding2):
    # Ensure both embeddings are 1-D arrays of type float64
    embedding1 = np.array(embedding1, dtype=np.float64).flatten()
    embedding2 = np.array(embedding2, dtype=np.float64).flatten()
    
    return distance.euclidean(embedding1, embedding2)

# Determine if embeddings match based on a threshold
def is_match(embedding1, embedding2, threshold=0.4):
    dist = compute_distance(embedding1, embedding2)
    return dist < threshold

def calculate_confidence(embedding1, embedding2, threshold=0.3):
    euclidean_distance = compute_distance(embedding1, embedding2)
    confidence_level = 100- euclidean_distance*10 
    return confidence_level


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

    # Convert embeddings to NumPy arrays
    embeddings = []
    for angle, stored_embedding in rows:
        # Parse stored embedding if it's in a JSON format or a delimited string
        try:
            if isinstance(stored_embedding, str):
                # Convert from JSON string to Python list and then to NumPy array
                embedding_array = np.array(json.loads(stored_embedding), dtype=np.float64)
            else:
                # If not a string, directly convert to NumPy array with proper type
                embedding_array = np.array(stored_embedding, dtype=np.float64)

            # Ensure it's a 1-D array
            embedding_array = embedding_array.flatten()

            embeddings.append((angle, embedding_array))
        except Exception as e:
            print(f"Error parsing embedding: {e}")
    
    return embeddings

# Search for the person in the database
def find_matching_person(new_embedding):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT person_id FROM face_embeddings")
    person_ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    # Convert new_embedding to a 1-D array if not already
    new_embedding = np.array(new_embedding, dtype=np.float64).flatten()

    for person_id in person_ids:
        embeddings = fetch_embeddings_from_db(person_id)
        for angle, stored_embedding in embeddings:
            # Compare using is_match function
            if is_match(new_embedding, stored_embedding):
                return person_id
    return None

def log_activity(person_id, confidence_level):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO activity_log (person_id, confidence_level)
        VALUES (%s, %s)
    """, (person_id, confidence_level))
    conn.commit()
    cursor.close()
    conn.close()

def run_access_control_script():
        script_path = "src/access_control.py"  
        subprocess.run(["python", script_path])

# Load models
detector, shape_predictor, face_rec_model = load_models()

# Main function
def main():

    # Load and process the input image
    input_image_path = "data/temp/temp.jpg" 
    image = cv2.imread(input_image_path)
    
    if image is None:
        print(f"Error loading image: {input_image_path}")
        return
    start_time = time.time()
    face_descriptors, _ = get_face_descriptor(image, detector, shape_predictor, face_rec_model)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time get_face_descriptor: {execution_time} seconds")
    if not face_descriptors:
        print("No faces detected in the image.")
        return
    
    # Assume the first detected face is the one we want to recognize
    new_embedding = face_descriptors[0]

    # Find matching person
    start_time = time.time()
    person_id = find_matching_person(new_embedding)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time db: {execution_time} seconds")
    if person_id:
        print(f"Face matched with person ID: {person_id}")
        confidence_level = calculate_confidence(new_embedding, np.array(new_embedding))  # Use a reference embedding for comparison
        log_activity(person_id, confidence_level)
    else:
        print("No matching faces found in the database.")
        run_access_control_script()
        


if __name__ == "__main__":
    main()
