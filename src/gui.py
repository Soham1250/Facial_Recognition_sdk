import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
from main import detect_faces, capture_face, save_face_image, load_models, run_face_recognition_script

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1280x720")
        self.root.configure(bg="#f7f7f7")

        # Video display frame
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(pady=20)

        # Control buttons
        self.start_button = ttk.Button(self.root, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(self.root, text="Stop Recognition", command=self.stop_recognition, state="disabled")
        self.stop_button.pack(pady=10)

        self.exit_button = ttk.Button(self.root, text="Exit", command=self.exit_program)
        self.exit_button.pack(pady=10)

        # Initialize variables
        self.cap = None
        self.running = False
        self.video_thread = None
        self.detector, self.shape_predictor, self.face_rec_model = load_models()  # Load models

    def start_recognition(self):
        # Start the video capture and recognition thread
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
        # Stop the recognition
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def exit_program(self):
        # Stop recognition and close the program
        self.stop_recognition()
        if self.cap is not None:
            self.cap.release()
        self.root.quit()

    def video_loop(self):
        # Video loop for capturing and displaying frames
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect faces and draw rectangles
            faces = detect_faces(frame)  # Using the detect_faces function from main.py
            for (x, y, w, h) in faces:
                face_image = capture_face(frame, (x, y, w, h))  # Crop the detected face
                image_path = save_face_image(face_image)  # Save the face temporarily
                run_face_recognition_script()  # Run recognition on the saved face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle

            # Convert frame to ImageTk format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.config(image=imgtk)

        # Release the camera when stopping
        if self.cap is not None:
            self.cap.release()
            self.video_frame.config(image="")

# Initialize GUI
root = tk.Tk()
app = FaceRecognitionGUI(root)
root.mainloop()