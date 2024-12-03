import PyInstaller.__main__
import os
import sys

def build_executable():
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    src_dir = os.path.join(current_dir, 'src')
    models_dir = os.path.join(current_dir, 'models')
    app_path = os.path.join(src_dir, 'facial_recognition_app.py')
    
    # Ensure the models directory exists
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        return
        
    # PyInstaller options
    options = [
        app_path,                          # Your main script
        '--name=FaceRecognition',          # Name of the executable
        '--onefile',                       # Create a single executable
        '--windowed',                      # Window mode (no console)
        '--clean',                         # Clean cache
        f'--add-data={models_dir};models', # Add models directory
        '--hidden-import=cv2',
        '--hidden-import=dlib',
        '--hidden-import=PIL',
        '--hidden-import=PIL.Image',
        '--hidden-import=PIL.ImageTk',
        '--hidden-import=numpy',
        '--hidden-import=mysql.connector',
        '--hidden-import=scipy',
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.messagebox',
        '--collect-data=dlib',             # Include dlib data files
        '--collect-data=cv2',              # Include OpenCV data files
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(options)

if __name__ == '__main__':
    build_executable()
