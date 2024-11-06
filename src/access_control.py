import cv2
import getpass 
import tkinter as tk
from tkinter import messagebox
import mysql.connector #type:ignore
import hashlib
import base64
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

def hash_password(password):
    sha256 = hashlib.sha256()
    sha256.update(password.encode('utf-8'))
    hash_bytes = sha256.digest()
    base64_encoded_hash = base64.b64encode(hash_bytes).decode('utf-8')
    return base64_encoded_hash

def check_password_in_db(password_hash):
    conn = connect_db()
    cursor = conn.cursor()
    
    query = "SELECT COUNT(*) FROM face_embeddings WHERE password = %s"
    cursor.execute(query, (password_hash,))
    count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    return count > 0

def on_submit():
    password = password_entry.get()
    password_hash = hash_password(password)
    
    if check_password_in_db(password_hash):
        messagebox.showinfo("Access Control", "Access granted.")
        window.destroy() 
    else:
        messagebox.showerror("Access Control", "Access denied. Incorrect password.")
        password_entry.delete(0, tk.END)

def show_password_window():
    global window, password_entry

    window = tk.Tk()
    window.title("Password Entry")

    tk.Label(window, text="Enter your password:").pack(padx=100, pady=50)
    
    password_entry = tk.Entry(window, show="*", width=30)
    password_entry.pack(padx=100, pady=10)

    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack(padx=10, pady=10)

    window.mainloop()

def main():
    print("Face not recognized from the database.")
    show_password_window()

if __name__ == "__main__":
    main()
