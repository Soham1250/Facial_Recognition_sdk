import cv2
import numpy as np #type:ignore
import mysql.connector #type:ignore
from scipy.spatial import distance #type:ignore
import dlib #type:ignore
import json  # Import the JSON module
import hashlib
import base64
<<<<<<< HEAD
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH, SAVE_PATH
=======
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH
>>>>>>> d7c03bb61e54695e59dedfb212ed1c365e5030d8

def connect_db():
    conn = mysql.connector.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    return conn

def hash_password(password):
    sha256 = hashlib.sha256()
    sha256.update(password.encode('utf-8'))
    hash_bytes = sha256.digest()
    base64_encoded_hash = base64.b64encode(hash_bytes).decode('utf-8')
    return base64_encoded_hash

def create_password(person_id: str) -> str:
    ans = ""
    for i in range(len(person_id) - 1, -1, -1):
        if person_id[i] == '_':
            break
        ans += person_id[i]
    return ans[::-1]+"@123"  # Reverse the string before returning



def insert_embedding(person_id, angle, embedding):
        conn = connect_db()
        cur = conn.cursor()
        # Convert the embedding (which is a numpy array or list) to a JSON string
        embedding_json = json.dumps(embedding.tolist())  # Convert to JSON string
        cur.execute("""
            INSERT INTO face_embeddings (person_id, angle, embedding,password)
            VALUES (%s, %s, %s,%s)
        """, (person_id, angle, embedding_json,hash_password(create_password(person_id))))  # Use the JSON string here
        conn.commit()
        cur.close()
        conn.close()


def extract_features(image, detector, shape_predictor, face_rec_model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None  # No faces detected

    features_list = []
    for face in faces:
        shape = shape_predictor(gray, face)
        descriptor = face_rec_model.compute_face_descriptor(image, shape)
        features_list.append(np.array(descriptor))
    
    return features_list

def load_models():
    SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
    FACE_REC_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
    return detector, shape_predictor, face_rec_model

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    detector, shape_predictor, face_rec_model = load_models()
    features = extract_features(image, detector, shape_predictor, face_rec_model)
    return features

# Example usage
<<<<<<< HEAD
front_image_path = 'data/sample_images/3028_Ketan_F.jpg'
left_image_path = 'data/sample_images/3028_Ketan_L.jpg'
right_image_path = 'data/sample_images/3028_Ketan_R.jpg'
=======
front_image_path = 'Path to Front face image'
left_image_path = 'Path to Left face image'
right_image_path = 'Path to Right face image'
>>>>>>> d7c03bb61e54695e59dedfb212ed1c365e5030d8

front_features = process_image(front_image_path)
left_features = process_image(left_image_path)
right_features = process_image(right_image_path)

# Insert embeddings into the database
<<<<<<< HEAD
person_id = '3028_Ketan'
=======
person_id = 'Person ID / Name'
>>>>>>> d7c03bb61e54695e59dedfb212ed1c365e5030d8
if front_features:
    for feature in front_features:
        insert_embedding(person_id, 'front', feature)
if left_features:
    for feature in left_features:
        insert_embedding(person_id, 'left', feature)
if right_features:
    for feature in right_features:
        insert_embedding(person_id, 'right', feature)
