import csv
import datetime
import queue
import time
import tkinter as tk
from tkinter import ttk
import cv2
import os
import numpy as np
import dlib
import mysql.connector
from mysql.connector import pooling, Error
from mysql.connector import pooling
from PIL import Image, ImageTk
import threading
import json
from scipy.spatial import distance
import time
from utils import DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT, SAVE_PATH, SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH
import logging
from contextlib import contextmanager
import time
import socket
import random
import sys

class DBConnectionManager:
    _instance = None
    _pool = None
    _last_connection_attempt = 0
    _backoff_time = 1  # Initial backoff time in seconds

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            cls._instance._initialize_pool()
            cls._instance._start_cleanup_thread()
        return cls._instance

    def _release_socket(self):
        """Release any existing socket connections"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.close()
        except Exception as e:
            logging.warning(f"Error releasing socket: {e}")

    def _wait_before_retry(self):
        """Implement exponential backoff with jitter"""
        sleep_time = self._backoff_time + random.uniform(0, 0.5)  # Add jitter
        time.sleep(sleep_time)
        self._backoff_time = min(self._backoff_time * 2, 60)  # Max backoff 60 seconds
        self._last_connection_attempt = time.time()

    def _check_connection_health(self, conn):
        """Check if connection is healthy"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False

    def _initialize_pool(self):
        """Initialize the connection pool with retry mechanism and socket management"""
        if self._pool is None:
            pool_config = {
                "pool_name": "face_recognition_pool",
                "pool_size": 5,
                "host": DB_HOST,
                "database": DB_NAME,
                "user": DB_USER,
                "password": DB_PASS,
                "port": DB_PORT,
                "pool_reset_session": True,
                "connect_timeout": 5,
                "use_pure": True,
                "autocommit": True,
            }

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    self._release_socket()
                    if retry_count > 0:
                        self._wait_before_retry()

                    self._pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)

                    # Verify pool health
                    test_conn = self._pool.get_connection()
                    if self._check_connection_health(test_conn):
                        test_conn.close()
                        logging.info("MySQL connection pool initialized successfully")
                        self._backoff_time = 1  # Reset backoff time on success
                        break
                    else:
                        raise Exception("Connection health check failed")

                except Error as err:
                    retry_count += 1
                    logging.warning(f"Attempt {retry_count} failed: {err}")
                    if retry_count == max_retries:
                        logging.error(f"Failed to initialize connection pool after {max_retries} attempts")
                        raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup and health check"""
        conn = None
        try:
            conn = self._pool.get_connection(wait_timeout=10)  # Wait up to 10 seconds
            if not self._check_connection_health(conn):
                logging.warning("Unhealthy connection detected, reinitializing pool")
                self._initialize_pool()
                conn = self._pool.get_connection()

            conn.cmd_reset_connection()
            yield conn

        except Error as err:
            logging.error(f"Database connection error: {err}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn:
                try:
                    if conn.is_connected():
                        conn.close()
                except Exception as e:
                    logging.error(f"Error closing connection: {e}")

    def _cleanup_idle_connections(self):
        """Clean up idle connections in the pool"""
        if self._pool:
            for conn in list(self._pool._cnx_queue):
                try:
                    if conn.is_connected():
                        conn.ping(reconnect=False, attempts=1, delay=0)
                    else:
                        conn.close()
                except Exception as e:
                    logging.warning(f"Error during idle connection cleanup: {e}")

    def _start_cleanup_thread(self):
        """Start a background thread to clean up idle connections"""
        def cleanup_task():
            while True:
                self._cleanup_idle_connections()
                time.sleep(300)  # Run every 5 minutes

        threading.Thread(target=cleanup_task, daemon=True).start()

    def cleanup(self):
        """Cleanup all connections in the pool"""
        if self._pool:
            for conn in self._pool._cnx_queue:
                try:
                    if conn.is_connected():
                        conn.close()
                except Exception:
                    pass
            self._pool = None
            self._release_socket()
            logging.info("Connection pool cleaned up")

    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()
    _instance = None
    _pool = None
    _last_connection_attempt = 0
    _backoff_time = 1  # Initial backoff time in seconds

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            cls._instance._initialize_pool()
            cls._instance._start_cleanup_thread()
        return cls._instance

    def _release_socket(self):
        """Release any existing socket connections"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.close()
        except Exception as e:
            logging.warning(f"Error releasing socket: {e}")

    def _wait_before_retry(self):
        """Implement exponential backoff with jitter"""
        sleep_time = self._backoff_time + random.uniform(0, 0.5)  # Add jitter
        time.sleep(sleep_time)
        self._backoff_time = min(self._backoff_time * 2, 60)  # Max backoff 60 seconds
        self._last_connection_attempt = time.time()

    def _check_connection_health(self, conn):
        """Check if connection is healthy"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False

    def _initialize_pool(self):
        """Initialize the connection pool with retry mechanism and socket management"""
        if self._pool is None:
            pool_config = {
                "pool_name": "face_recognition_pool",
                "pool_size": 5,
                "host": DB_HOST,
                "database": DB_NAME,
                "user": DB_USER,
                "password": DB_PASS,
                "port": DB_PORT,
                "pool_reset_session": True,
                "connect_timeout": 5,
                "use_pure": True,
                "autocommit": True,
            }

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    self._release_socket()
                    if retry_count > 0:
                        self._wait_before_retry()

                    self._pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)

                    # Verify pool health
                    test_conn = self._pool.get_connection()
                    if self._check_connection_health(test_conn):
                        test_conn.close()
                        logging.info("MySQL connection pool initialized successfully")
                        self._backoff_time = 1  # Reset backoff time on success
                        break
                    else:
                        raise Exception("Connection health check failed")

                except Error as err:
                    retry_count += 1
                    logging.warning(f"Attempt {retry_count} failed: {err}")
                    if retry_count == max_retries:
                        logging.error(f"Failed to initialize connection pool after {max_retries} attempts")
                        raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup and health check"""
        conn = None
        try:
            conn = self._pool.get_connection()
            if not self._check_connection_health(conn):
                logging.warning("Unhealthy connection detected, reinitializing pool")
                self._initialize_pool()
                conn = self._pool.get_connection()

            conn.cmd_reset_connection()
            yield conn

        except Error as err:
            logging.error(f"Database connection error: {err}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn:
                try:
                    if conn.is_connected():
                        conn.close()
                except Exception as e:
                    logging.error(f"Error closing connection: {e}")

    def _cleanup_idle_connections(self):
        """Clean up idle connections in the pool"""
        if self._pool:
            for conn in list(self._pool._cnx_queue):
                try:
                    if conn.is_connected():
                        conn.ping(reconnect=False, attempts=1, delay=0)
                    else:
                        conn.close()
                except Exception as e:
                    logging.warning(f"Error during idle connection cleanup: {e}")

    def _start_cleanup_thread(self):
        """Start a background thread to clean up idle connections"""
        def cleanup_task():
            while True:
                self._cleanup_idle_connections()
                time.sleep(300)  # Run every 5 minutes

        threading.Thread(target=cleanup_task, daemon=True).start()

    def cleanup(self):
        """Cleanup all connections in the pool"""
        if self._pool:
            for conn in self._pool._cnx_queue:
                try:
                    if conn.is_connected():
                        conn.close()
                except Exception:
                    pass
            self._pool = None
            self._release_socket()
            logging.info("Connection pool cleaned up")

    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()
    _instance = None
    _pool = None
    _last_connection_attempt = 0
    _backoff_time = 1  # Initial backoff time in seconds
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnectionManager, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance
    
    def _release_socket(self):
        """Release any existing socket connections"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.close()
        except:
            pass

    def _wait_before_retry(self):
        """Implement exponential backoff with jitter"""
        current_time = time.time()
        time_since_last_attempt = current_time - self._last_connection_attempt
        
        if time_since_last_attempt < self._backoff_time:
            sleep_time = self._backoff_time - time_since_last_attempt
            # Add jitter to prevent thundering herd
            sleep_time += random.uniform(0, 1)
            time.sleep(sleep_time)
        
        self._backoff_time = min(self._backoff_time * 2, 60)  # Max backoff of 60 seconds
        self._last_connection_attempt = time.time()

    def _check_connection_health(self, conn):
        """Check if connection is healthy"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except:
            return False
    
    def _initialize_pool(self):
        """Initialize the connection pool with retry mechanism and socket management"""
        if self._pool is None:
            pool_config = {
                "pool_name": "face_recognition_pool",
                "pool_size": 5,
                "host": DB_HOST,
                "database": DB_NAME,
                "user": DB_USER,
                "password": DB_PASS,
                "port": DB_PORT,
                "pool_reset_session": True,
                "connect_timeout": 5,
                "use_pure": True,
                "autocommit": True,
                "get_warnings": True,
                "raise_on_warnings": True,
                "connection_timeout": 10,
                "pool_reset_session": True
            }
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Release any existing connections
                    self._release_socket()
                    
                    # Wait before retry using exponential backoff
                    if retry_count > 0:
                        self._wait_before_retry()
                    
                    self._pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)
                    
                    # Verify pool health
                    test_conn = self._pool.get_connection()
                    if self._check_connection_health(test_conn):
                        test_conn.close()
                        logging.info("MySQL connection pool initialized successfully")
                        self._backoff_time = 1  # Reset backoff time on success
                        break
                    else:
                        raise Exception("Connection health check failed")
                        
                except Error as err:
                    retry_count += 1
                    if retry_count == max_retries:
                        logging.error(f"Failed to initialize connection pool after {max_retries} attempts: {err}")
                        raise
                    logging.warning(f"Attempt {retry_count} failed, retrying...")
                finally:
                    self._release_socket()
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup and health check"""
        conn = None
        try:
            conn = self._pool.get_connection()
            if not self._check_connection_health(conn):
                # If connection is unhealthy, try to reinitialize pool
                self._pool = None
                self._initialize_pool()
                conn = self._pool.get_connection()
            
            # Reset session to ensure clean state
            conn.cmd_reset_connection()
            yield conn
            
        except Error as err:
            logging.error(f"Database connection error: {err}")
            if conn:
                try:
                    conn.rollback()  # Rollback any pending transactions
                except:
                    pass
            raise
        finally:
            if conn is not None:
                try:
                    conn.cmd_reset_connection()  # Reset connection state
                    if conn.is_connected():
                        conn.close()
                except:
                    pass
    
    def cleanup(self):
        """Cleanup all connections in the pool"""
        if self._pool is not None:
            for conn in self._pool._cnx_queue:
                try:
                    if conn.is_connected():
                        conn.close()
                except:
                    pass
            self._pool = None
            self._release_socket()
            logging.info("Connection pool cleaned up")
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()

class FaceRecognition:
    _db_manager = None
    _last_recognized_user = None  # Track the last recognized user

    def __init__(self):
        if FaceRecognition._db_manager is None:
            FaceRecognition._db_manager = DBConnectionManager()

    @staticmethod
    def get_connection():
        return DBConnectionManager().get_connection()


    @staticmethod
    def initialize_embeddings_dict():
        """Fetch all embeddings from the database and store them in a dictionary."""
        FaceRecognition._embeddings_dict = {}
        with FaceRecognition.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT person_id, embedding FROM face_embeddings")
            rows = cursor.fetchall()
            cursor.close()

        for person_id, embedding in rows:
            try:
                embedding_array = np.array(json.loads(embedding), dtype=np.float64)
                if person_id not in FaceRecognition._embeddings_dict:
                    FaceRecognition._embeddings_dict[person_id] = []
                FaceRecognition._embeddings_dict[person_id].append(embedding_array)
            except Exception as e:
                print(f"Error parsing embedding for person_id {person_id}: {e}")

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
    def find_matching_person(new_embedding):
        """Find a matching person_id using the dictionary instead of database calls."""
        new_embedding = np.array(new_embedding, dtype=np.float64).flatten()

        for person_id, embeddings in FaceRecognition._embeddings_dict.items():
            for stored_embedding in embeddings:
                if FaceRecognition.is_match(new_embedding, stored_embedding):
                    return person_id
        return None

    @staticmethod
    def fetch_embeddings_from_db(person_id):
        with FaceRecognition.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT angle, embedding FROM face_embeddings WHERE person_id = %s
            """, (person_id,))
            rows = cursor.fetchall()
            cursor.close()

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
    def log_activity(person_id, confidence_level):
        with FaceRecognition.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO activity_log (person_id, confidence_level)
                VALUES (%s, %s)
            """, (person_id, confidence_level))
            conn.commit()
            cursor.close()


    _db_manager = DBConnectionManager()
    
    @staticmethod
    def get_connection():
        return DBConnectionManager().get_connection()
    
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
        with FaceRecognition.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT angle, embedding FROM face_embeddings WHERE person_id = %s
            """, (person_id,))
            rows = cursor.fetchall()
            cursor.close()

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
        with FaceRecognition.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT person_id FROM face_embeddings")
            person_ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
        
        new_embedding = np.array(new_embedding, dtype=np.float64).flatten()

        for person_id in person_ids:
            embeddings = FaceRecognition.fetch_embeddings_from_db(person_id)
            for angle, stored_embedding in embeddings:
                if FaceRecognition.is_match(new_embedding, stored_embedding):
                    return person_id
        return None

    @staticmethod
    def log_activity(person_id, confidence_level):
        with FaceRecognition.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO activity_log (person_id, confidence_level)
                VALUES (%s, %s)
            """, (person_id, confidence_level))
            conn.commit()
            cursor.close()


    @staticmethod
    def add_embedding_to_dict(person_id, embedding):
        """Add a new embedding to the dictionary after adding it to the database."""
        embedding_array = np.array(json.loads(embedding), dtype=np.float64)
        if person_id not in FaceRecognition._embeddings_dict:
            FaceRecognition._embeddings_dict[person_id] = []
        FaceRecognition._embeddings_dict[person_id].append(embedding_array)


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
        
        # Show the captured image before recognition


        face_descriptors, _ = FaceRecognition.get_face_descriptor(
            image, self.detector, self.shape_predictor, self.face_rec_model
        )
        
        if not face_descriptors:
            print("No faces detected in the image.")
            return False

        total_scans += 1
        new_embedding = face_descriptors[0]
        person_id = FaceRecognition.find_matching_person(new_embedding)
        
        if person_id:
            successful_scans += 1
            self.show_captured_image(image_path, parent)
            
            if person_id != FaceRecognition._last_recognized_user:
                # Show the popup
                FaceRecognition._last_recognized_user = person_id

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

                # Add an OK button
                ok_button = ttk.Button(popup, text="OK", command=popup.destroy)
                ok_button.pack()
                
                # Create frame for buttons
                button_frame = ttk.Frame(popup)
                button_frame.pack(pady=10)

                # Add "Not you?" button
                def open_feedback():
                    popup.destroy()  # Close the popup
                    feedback_window = FeedbackWindow()
                    feedback_window.show()

                not_you_button = ttk.Button(button_frame, text="Not you?", command=open_feedback)
                not_you_button.pack(side=tk.LEFT, padx=10)

                # Make the popup modal
                popup.transient(parent)
                popup.grab_set()
                popup.wait_window()

                # Delay for 10 seconds
                print(f"Waiting 5 seconds after recognizing {person_id}...")
                time.sleep(5)  # Add the delay here
            else:
                print(f"User {person_id} already recognized recently.")
        return False
    

    def show_captured_image(self, image_path, parent):
        """Displays the captured image in a popup window and closes it after a duration."""
        captured_image_popup = tk.Toplevel(parent)
        captured_image_popup.title("Captured Image")
        
        # Load the image
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Resize for display
        imgtk = ImageTk.PhotoImage(img)
        
        # Display the image
        img_label = ttk.Label(captured_image_popup, image=imgtk)
        img_label.image = imgtk  # Keep a reference to avoid garbage collection
        img_label.pack()
        
        # Set the duration for how long the popup will be displayed (in milliseconds)
        duration = 2500 # 5 seconds
        
        # Schedule the popup to close after the duration
        captured_image_popup.after(duration, captured_image_popup.destroy)
        
        # Make the popup modal
        captured_image_popup.transient(parent)
        captured_image_popup.grab_set()
        captured_image_popup.wait_window()



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
            "Krishna Naik",
            "Krutik Mandre",
            "M Jayadevi",
            "Mahesh S. Bhoop",
            "Mannu Vishwakarma",
            "Manoj Yadav",
            "Omkar Nikam",
            "Pritesh Mistry",
            "Priti Gaikwad",
            "Roopnarayan Gupta",
            "Sagar Tondvalkar",
            "Sachin Patil",
            "Sandesh Kurtdkar",
            "Shyamal Mishra",
            "Soham Pansare",
            "Sonal Mayekar",
            "Sushil Khetre",
            "Vaibhav Pawar",
            "Vikrant Sawant"]
        
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
        selected_employee = self.employee_combo.get()  # Get the selected employee name
        if selected_employee:
            try:
                # Connect to the database using DBConnectionManager
                with FaceRecognition.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Increment the Frequency column for the selected employee
                    query = "UPDATE employee_selection SET Frequency = Frequency + 1 WHERE Name = %s"
                    cursor.execute(query, (selected_employee,))
                    conn.commit()  # Commit the transaction
                    
                    # Show success message to the user
                    from tkinter import messagebox
                    messagebox.showinfo("Thank you correction!")
            except Exception as e:
                # Show error message in case of any database issue
                from tkinter import messagebox
                messagebox.showerror("Error", f"Failed to update frequency: {e}")
            finally:
                # Close the feedback window
                self.app.destroy()
        else:
            # Show a warning if no employee is selected
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please select an employee name.")

    def show(self):
        self.app.mainloop()

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition")
        self.geometry("1280x720")

        # Initialize attributes
        self.last_scan_time = 0  # Initialize the last scan time
        self.total_scans = 0
        self.successful_scans = 0
        self.current_user= None

        # Use grid layout for main window
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Left Frame for Live Feed and Controls
        self.left_frame = tk.Frame(self, bg="black")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.live_feed_label = tk.Label(self.left_frame, text="Live Feed", bg="black", fg="white", font=("Helvetica", 16))
        self.live_feed_label.pack(expand=True, fill="both", pady=10)

        self.start_button = ttk.Button(self.left_frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(side="left", padx=10)

        self.stop_button = ttk.Button(self.left_frame, text="Stop Recognition", command=self.stop_recognition)
        self.stop_button.pack(side="left", padx=10)

        # self.exit_button = ttk.Button(self.left_frame, text="Exit", command=self.exit_program)
        # self.exit_button.pack(side="left", padx=10)

        # Right Frame for Recognition Details
        self.right_frame = tk.Frame(self, bg="white")   
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.captured_image_label = tk.Label(self.right_frame, text="Captured Image", bg="gray", fg="white", font=("Helvetica", 16))
        self.captured_image_label.pack(pady=10, fill="both", expand=True)

        self.last_user_label = ttk.Label(self.right_frame, text="Last Recognized User: None", font=("Helvetica", 12))
        self.last_user_label.pack(pady=5)

        self.current_user_label = ttk.Label(self.right_frame, text="Current Recognized User: None", font=("Helvetica", 12))
        self.current_user_label.pack(pady=5)

        # Add a timestamp label for the last scan
        self.last_scan_timestamp_label = ttk.Label(self.right_frame, text="Last Scan: N/A", font=("Helvetica", 12))
        self.last_scan_timestamp_label.pack(pady=5)

        # Add an accuracy label to avoid missing attributes
        self.accuracy_label = ttk.Label(self.right_frame, text="Accuracy: N/A", font=("Helvetica", 12))
        self.accuracy_label.pack(pady=5)

        # Frame for "Not You?" and "OK" buttons
        self.button_frame = tk.Frame(self.right_frame, bg="white")
        self.button_frame.pack(pady=10)
        
        # Add "OK" button to the right 
        self.ok_button = ttk.Button(self.button_frame, text="OK", command=self.confirm_identity)
        self.ok_button.pack(side="left", padx=10)

        # Add "Not You?" button
        self.not_you_button = ttk.Button(self.button_frame, text="Not You?", command=self.not_you_feedback)
        self.not_you_button.pack(side="left", padx=10)


        # Initialize variables
        self.cap = None
        self.running = False
        self.live_feed_thread = None
        self.recognition_thread = None
        self.camera_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=10)
        self.last_scan_time

        # Initialize CoreFunctions (fix missing attribute 'core')
        self.core = CoreFunctions()

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")

            # Initialize confirmation flag
            self.waiting_for_confirmation = False

            FaceRecognition.initialize_embeddings_dict()

            # Start live feed thread
            self.live_feed_thread = threading.Thread(target=self.run_live_feed, daemon=True)
            self.live_feed_thread.start()

            # Start recognition thread
            self.recognition_thread = threading.Thread(target=self.run_recognition, daemon=True)
            self.recognition_thread.start()

    def stop_recognition(self):
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        with self.camera_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
           
        FaceRecognition._last_recognized_user = None

        # Log session to CSV
        SessionLogger.log_session(
            SAVE_PATH,
            self.total_scans,
            self.successful_scans,
            self.current_user,
            confidence= 100 # You might want to pass actual confidence if tracked
        )


        
    def confirm_identity(self):
        """Confirm the identity when the 'OK' button is clicked."""
        self.waiting_for_confirmation = False

    def run_live_feed(self):
        """Continuously capture frames from the camera and update the live feed."""
        with self.camera_lock:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Unable to access the camera.")
                return

        while self.running:
            with self.camera_lock:
                ret, frame = self.cap.read()
                if not ret:
                    break

            # Convert frame to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update live feed asynchronously
            self.after(0, self.update_live_feed_label, imgtk)

            # Push frame to the queue without waiting
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # If the queue is full, drop the frame to avoid lag

        with self.camera_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def update_live_feed_label(self, imgtk):
        """Update the live feed label without blocking the main thread."""
        self.live_feed_label.imgtk = imgtk  # Store reference to avoid garbage collection
        self.live_feed_label.config(image=imgtk)

    def update_captured_image(self,frame):
        """Update the captured image label."""
        temp_image_path = os.path.join(SAVE_PATH, "temp.jpg")
        try:
            img = Image.open(temp_image_path)
            img = img.resize((300, 300))  # Resize for display
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the captured image label
            self.captured_image_label.imgtk = imgtk  # Store a reference to avoid garbage collection
            self.captured_image_label.config(image=imgtk)
        except Exception as e:
            print(f"Error loading captured image: {e}")

    def run_recognition(self):
        """Continuously process frames for face recognition."""
       

        while self.running:
            try:
                # Retrieve a frame from the queue
                frame = self.frame_queue.get(timeout=1)

                # Check if 10 seconds have passed since the last scan
                current_time = time.time()
                if current_time - self.last_scan_time < 10:
                    continue  # Skip recognition if 10 seconds haven't passed

                # Perform recognition logic
                face_descriptors, faces = FaceRecognition.get_face_descriptor(
                    frame, self.core.detector, self.core.shape_predictor, self.core.face_rec_model
                )

                if face_descriptors:
                    self.total_scans += 1

                    # Save the first detected face and update captured image
                    face = faces[0]
                    (x, y, w, h) = face.left(), face.top(), face.width(), face.height()
                    face_image = frame[y:y + h, x:x + w]
                    self.core.save_face_image(face_image)
                    self.update_captured_image(face_image)

                    # Perform recognition
                    for descriptor in face_descriptors:
                        self.current_user = FaceRecognition.find_matching_person(descriptor)

                        # If the current user differs from the last user, wait for confirmation
                        if self.current_user and self.current_user != FaceRecognition._last_recognized_user:
                            self.waiting_for_confirmation = True
                            self.current_user_label.config(
                                text=f"Current Recognized User: {self.current_user} (Confirm with OK)"
                            )
                            FaceRecognition._last_recognized_user = self.current_user

                            # Wait for "OK" button press before proceeding
                            while self.waiting_for_confirmation and self.running:
                                time.sleep(0.1)

                        # If confirmed, update the timestamp and labels
                        if self.current_user:
                            self.successful_scans += 1
                            self.last_user_label.config(
                                text=f"Last Recognized User: {FaceRecognition._last_recognized_user}"
                            )
                            self.current_user_label.config(
                                text=f"Current Recognized User: {self.current_user}"
                            )

                            # Update last scan time and timestamp
                            self.last_scan_time = current_time
                            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                            self.last_scan_timestamp_label.config(text=f"Last Scan: {formatted_time}")

                    # Update accuracy
                    if self.total_scans > 0:
                        accuracy = (self.successful_scans / self.total_scans) * 100
                        self.accuracy_label.config(text=f"Accuracy: {self.successful_scans:.2f} / {self.total_scans:.2f} = {accuracy:.2f}%")

            except queue.Empty:
                continue  # No frames available, keep waiting
            except Exception as e:
                print(f"Error in recognition thread: {e}")



    def not_you_feedback(self):
        """Use existing Not You? button logic."""
        feedback_window = FeedbackWindow()
        feedback_window.show()


class SessionLogger:
    @staticmethod
    def log_session(temp_folder, total_scans, successful_scans, recognized_user, confidence):
        """
        Log session information to a CSV file in the temp folder
        
        Args:
            temp_folder (str): Path to temporary folder
            total_scans (int): Total number of face scans
            successful_scans (int): Number of successful recognitions
            recognized_user (str): Identified user
            confidence (float): Confidence level of recognition
        """
        # Create temp folder if it doesn't exist
        os.makedirs(temp_folder, exist_ok=True)
        
        # Calculate accuracy
        accuracy = (successful_scans / total_scans * 100) if total_scans > 0 else 0
        
        # Generate unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Use underscores or omit colons
        file_path = os.path.join(SAVE_PATH, f"session_stats_{timestamp}.csv")
        
        # CSV headers
        headers = ['Timestamp', 'Total Scans', 'Successful Scans', 'Accuracy', 'Recognized User', 'Confidence']
        
        # Write to CSV
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerow([
                timestamp, 
                total_scans, 
                successful_scans, 
                f"{accuracy:.2f}%", 
                recognized_user or "No Match", 
                f"{confidence:.2f}" if confidence is not None else "N/A"
            ])
        
        return file_path


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
