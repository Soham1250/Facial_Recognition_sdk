import json
import cv2
import dlib  # type: ignore
import numpy as np  # type: ignore
import os, sys, time, subprocess
import mysql.connector  # type: ignore
from scipy.spatial import distance  # type: ignore
import tkinter as tk
from tkinter import ttk
import subprocess
import time
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH


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
    embedding1 = np.array(embedding1, dtype=np.float64).flatten()
    embedding2 = np.array(embedding2, dtype=np.float64).flatten()
    return distance.euclidean(embedding1, embedding2)

# Determine if embeddings match based on a threshold
def is_match(embedding1, embedding2, threshold=0.4):
    dist = compute_distance(embedding1, embedding2)
    return dist < threshold

def calculate_confidence(embedding1, embedding2, threshold=0.3):
    euclidean_distance = compute_distance(embedding1, embedding2)
    confidence_level = 100 - euclidean_distance * 10 
    return confidence_level

def run_access_control_script():
    script_path = "src/report.py"  
    subprocess.run(["python", script_path])

# Function to display pop-up using Tkinter with styling
def show_match_popup(person_id):
    popup = tk.Tk()
    popup.title("Face Match Found")
    popup.geometry("400x300")
    popup.configure(bg="#f7f7f7")

    header_frame = tk.Frame(popup, bg="#4a90e2")
    header_frame.grid(row=0, column=0, sticky="nsew")
    header_label = tk.Label(
        header_frame,
        text="Match Found!",
        font=("Helvetica", 18, "bold"),
        bg="#4a90e2",
        fg="white"
    )
    header_label.pack(pady=10, padx=10)

    popup.grid_rowconfigure(0, weight=1)
    popup.grid_rowconfigure(1, weight=3)
    popup.grid_rowconfigure(2, weight=1)
    popup.grid_rowconfigure(3, weight=1)
    popup.grid_rowconfigure(4, weight=1)
    popup.grid_columnconfigure(0, weight=1)

    label_text = f"Person ID: {person_id}"
    label = tk.Label(
        popup, 
        text=label_text, 
        font=("Helvetica", 14), 
        bg="#f7f7f7", 
        fg="#2c3e50"
    )
    label.grid(row=1, column=0, padx=10, pady=(20, 10))

    style = ttk.Style()
    style.configure(
        "Accent.TButton",
        font=("Helvetica", 12),
        padding=8,
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#357ABD")]
    )

    ok_button = ttk.Button(popup, text="OK", command=popup.destroy)
    ok_button.grid(row=2, column=0, pady=10)

    not_you_button = ttk.Button(popup, text="Not you?", command=run_access_control_script)
    not_you_button.grid(row=4, column=0, pady=(5, 20))

    popup.mainloop()

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

    embeddings = []
    for angle, stored_embedding in rows:
        try:
            if isinstance(stored_embedding, str):
                embedding_array = np.array(json.loads(stored_embedding), dtype=np.float64)
            else:
                embedding_array = np.array(stored_embedding, dtype=np.float64)
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
    
    new_embedding = np.array(new_embedding, dtype=np.float64).flatten()

    for person_id in person_ids:
        embeddings = fetch_embeddings_from_db(person_id)
        for angle, stored_embedding in embeddings:
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

def recognize_face(image_path, total_scans, successful_scans):
    detector, shape_predictor, face_rec_model = load_models()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error loading image: {image_path}")
        return False
    
    face_descriptors, _ = get_face_descriptor(image, detector, shape_predictor, face_rec_model)
    if not face_descriptors:
        print("No faces detected in the image.")
        return False

    total_scans[0] += 1  # Increment total scans
    new_embedding = face_descriptors[0]
    person_id = find_matching_person(new_embedding)
    
    if person_id:
        print(f"Face matched with person ID: {person_id}")
        confidence_level = calculate_confidence(new_embedding, np.array(new_embedding))
        log_activity(person_id, confidence_level)
        successful_scans[0] += 1  # Increment successful scans
        show_match_popup(person_id)
        return True
    else:
        print("No matching faces found in the database.")
        return False
