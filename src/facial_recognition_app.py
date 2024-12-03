import tkinter as tk
from tkinter import ttk
import cv2
import os
import numpy as np
import dlib
import mysql.connector
from PIL import Image, ImageTk
import threading
import json
from scipy.spatial import distance
import subprocess
import time
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH, SAVE_PATH

class FaceRecognition:
    @staticmethod
    def connect_db():
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        return conn

    @staticmethod
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

    @staticmethod
    def compute_distance(embedding1, embedding2):
        embedding1 = np.array(embedding1, dtype=np.float64).flatten()
        embedding2 = np.array(embedding2, dtype=np.float64).flatten()
        return distance.euclidean(embedding1, embedding2)

    @staticmethod
    def is_match(embedding1, embedding2, threshold=0.4):
        dist = FaceRecognition.compute_distance(embedding1, embedding2)
        return dist < threshold

    @staticmethod
    def calculate_confidence(embedding1, embedding2, threshold=0.3):
        euclidean_distance = FaceRecognition.compute_distance(embedding1, embedding2)
        confidence_level = 100 - euclidean_distance * 10 
        return confidence_level

    @staticmethod
    def fetch_embeddings_from_db(person_id):
        conn = FaceRecognition.connect_db()
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

    @staticmethod
    def find_matching_person(new_embedding):
        conn = FaceRecognition.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT person_id FROM face_embeddings")
        person_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        new_embedding = np.array(new_embedding, dtype=np.float64).flatten()

        for person_id in person_ids:
            embeddings = FaceRecognition.fetch_embeddings_from_db(person_id)
            for angle, stored_embedding in embeddings:
                if FaceRecognition.is_match(new_embedding, stored_embedding):
                    return person_id
        return None

    @staticmethod
    def log_activity(person_id, confidence_level):
        conn = FaceRecognition.connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO activity_log (person_id, confidence_level)
            VALUES (%s, %s)
        """, (person_id, confidence_level))
        conn.commit()
        cursor.close()
        conn.close()

    @staticmethod
    def show_match_popup(person_id, parent):
        popup = tk.Toplevel(parent)
        popup.title("Match Found")
        popup.geometry("300x150")
        
        # Center the popup
        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add message
        message = ttk.Label(popup, text=f"Match found: {person_id}", font=("Helvetica", 12))
        message.pack(pady=20)
        
        # Create frame for buttons
        button_frame = ttk.Frame(popup)
        button_frame.pack(pady=10)
        
        # Add OK button
        ok_button = ttk.Button(button_frame, text="OK", command=popup.destroy)
        ok_button.pack(side=tk.LEFT, padx=10)
        
        # Add "Not you?" button
        def open_feedback():
            popup.destroy()  # Close the popup
            feedback_window = FeedbackWindow()
            feedback_window.show()
        
        not_you_button = ttk.Button(button_frame, text="Not you?", command=open_feedback)
        not_you_button.pack(side=tk.LEFT, padx=10)
        
        # Make popup modal
        popup.transient(parent)
        popup.grab_set()
        popup.wait_window()

class CoreFunctions:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.detector, self.shape_predictor, self.face_rec_model = self.load_models()

    def load_models(self):
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
        return detector, shape_predictor, face_rec_model

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def save_face_image(self, face_image):
        face_image_path = os.path.join(SAVE_PATH, "temp.jpg")
        try:
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            cv2.imwrite(face_image_path, face_image)
            return face_image_path
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def capture_face(self, frame, face_coords):
        (x, y, w, h) = face_coords
        face_image = frame[y:y + h, x:x + w]
        return face_image

    def recognize_face(self, image_path, total_scans, successful_scans, parent):
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            return False
        
        face_descriptors, _ = FaceRecognition.get_face_descriptor(
            image, self.detector, self.shape_predictor, self.face_rec_model
        )
        
        if not face_descriptors:
            print("No faces detected in the image.")
            return False

        total_scans[0] += 1
        new_embedding = face_descriptors[0]
        person_id = FaceRecognition.find_matching_person(new_embedding)
        
        if person_id:
            print(f"Face matched with person ID: {person_id}")
            successful_scans[0] += 1
            FaceRecognition.show_match_popup(person_id, parent)
            return True
        return False

class FeedbackWindow:
    def __init__(self):
        self.app = tk.Tk()
        self.app.title("Employee Selection")
        self.app.geometry("600x400")
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.app, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add a label with larger font
        ttk.Label(main_frame, text="Select your name", 
                 font=("Helvetica", 16)).grid(row=0, column=0, pady=(40, 10))
        
        # Employee names from report.py
        self.employee_names = ["Ellaiah Gangadhari",
            "Gautam Binniwale",
            "Kishor B. Patil",
            "Krutik Mandre",
            "Krishna Naik",
            "M Jayadevi",
            "Mahesh S. Bhoop",
            "Mannu Vishwakarma",
            "Manoj  Yadav",
            "Pritesh Mistry",
            "Priti Gaikwad",
            "Roopnarayan Gupta",
            "Sachin Patil",
            "Sagar  Tondvalkar",
            "Sandesh Kurtdkar",
            "Shyamal Mishra",
            "Sonal Mayekar",
            "Sushil  Khetre",
            "Vaibhav Pawar",
            "Vikrant Sawant",
            "Omkar Nikam",
            "Soham Pansare"]
        
        # Add the combobox for employee selection
        self.selected_name = tk.StringVar()
        
        # Style the combobox
        style = ttk.Style()
        style.configure('Custom.TCombobox', padding=5)
        
        self.employee_combo = ttk.Combobox(main_frame, 
                                         textvariable=self.selected_name,
                                         values=self.employee_names,
                                         width=40,
                                         style='Custom.TCombobox',
                                         state='readonly')  # Make it readonly
        self.employee_combo.set("")  # Set default text
        self.employee_combo.grid(row=1, column=0, pady=20)
        
        # Add the submit button with custom styling
        submit_button = ttk.Button(main_frame, 
                                 text="Submit",
                                 command=self.submit_selection,
                                 width=20)
        submit_button.grid(row=2, column=0, pady=30)
        
        # Configure grid weights for centering
        self.app.grid_rowconfigure(0, weight=1)
        self.app.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Center the window
        self.app.update_idletasks()
        width = self.app.winfo_width()
        height = self.app.winfo_height()
        x = (self.app.winfo_screenwidth() // 2) - (width // 2)
        y = (self.app.winfo_screenheight() // 2) - (height // 2)
        self.app.geometry(f"{width}x{height}+{x}+{y}")

    def submit_selection(self):
        selected_employee = self.employee_combo.get()  # Get directly from combobox
        if selected_employee:
            print(f"Selected Employee: {selected_employee}")  # Print the selected name to the terminal
            
            try:
                # Connect to the database
                conn = FaceRecognition.connect_db()
                cursor = conn.cursor()
                
                # Update frequency for the selected employee
                query = "UPDATE employee_selection SET Frequency = Frequency + 1 WHERE Name = %s"
                cursor.execute(query, (selected_employee,))
                rows_affected = cursor.rowcount
                conn.commit()  # Commit the transaction to save changes
                
                if rows_affected > 0:
                    print(f"Updated frequency for {selected_employee}")
                else:
                    print(f"No record found for {selected_employee}")
                
            except Exception as e:
                print(f"Database error: {e}")
            finally:
                # Close the database connection
                if 'cursor' in locals():
                    cursor.close()
                if 'conn' in locals():
                    conn.close()
                
                # Close the Tkinter window
                self.app.destroy()
        else:
            print("Please select an employee name")

    def show(self):
        self.app.mainloop()

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition")
        self.geometry("1280x720")
        self.configure(bg="#f7f7f7")

        # Initialize CoreFunctions
        self.core = CoreFunctions()

        # Video display frame
        self.video_frame = tk.Label(self)
        self.video_frame.pack(pady=20)

        # Control buttons
        self.start_button = ttk.Button(self, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(self, text="Stop Recognition", command=self.stop_recognition, state="disabled")
        self.stop_button.pack(pady=10)

        self.exit_button = ttk.Button(self, text="Exit", command=self.exit_program)
        self.exit_button.pack(pady=10)

        # Label to display accuracy calculation and result
        self.accuracy_label = tk.Label(self, text="", font=("Helvetica", 14), bg="#f7f7f7", fg="#2c3e50")
        self.accuracy_label.pack(pady=20)

        # Initialize variables
        self.cap = None
        self.running = False
        self.video_thread = None
        self.total_scans = [0]
        self.successful_scans = [0]

    def start_recognition(self):
        # Reset counters at the start
        self.total_scans[0] = 0
        self.successful_scans[0] = 0
        self.accuracy_label.config(text="")

        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Unable to access the camera.")
                return
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.start()

    def stop_recognition(self):
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        if self.total_scans[0] > 0:
            accuracy = (self.successful_scans[0] / self.total_scans[0]) * 100
            calculation_text = f"Accuracy: {self.successful_scans[0]} / {self.total_scans[0]} = {accuracy:.2f}%"
        else:
            accuracy = 0.0
            calculation_text = "No scans performed."

        self.accuracy_label.config(text=calculation_text)

    def exit_program(self):
        self.stop_recognition()
        if self.cap is not None:
            self.cap.release()
        self.quit()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            faces = self.core.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_image = self.core.capture_face(frame, (x, y, w, h))
                image_path = self.core.save_face_image(face_image)
                self.core.recognize_face(image_path, self.total_scans, self.successful_scans, self)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.config(image=imgtk)

        if self.cap is not None:
            self.cap.release()
            self.video_frame.config(image="")

if __name__ == "__main__":
    app = GUI()
    app.mainloop()
