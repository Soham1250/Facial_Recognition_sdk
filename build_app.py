import PyInstaller.__main__
import os
import sys
import shutil

def clean_build_folders():
    """Clean build and dist folders"""
    folders = ['build', 'dist']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    # Remove spec file
    spec_file = 'FaceRecognition.spec'
    if os.path.exists(spec_file):
        os.remove(spec_file)

def build_app():
    """Build the application"""
    # Clean previous builds
    clean_build_folders()
    
    # Get the absolute path to the shape predictor and face recognition model
    base_path = os.path.dirname(os.path.abspath(__file__))
    shape_predictor = os.path.join(base_path, 'models', 'shape_predictor_68_face_landmarks.dat')
    face_rec_model = os.path.join(base_path, 'models', 'dlib_face_recognition_resnet_model_v1.dat')
    
    # Define PyInstaller arguments
    args = [
        'src/facial_recognition_app.py',  # Your main script
        '--name=FaceRecognition',         # Name of the executable
        '--onefile',                      # Create a single executable
        '--windowed',                     # Windows only: hide the console
        f'--add-data={shape_predictor};models',  # Include shape predictor model
        f'--add-data={face_rec_model};models',   # Include face recognition model
        '--hidden-import=mysql.connector.locales',
        '--hidden-import=mysql.connector.locales.eng',
        '--hidden-import=mysql.connector.plugins',
        '--hidden-import=mysql.connector.plugins.mysql_native_password',
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=dlib',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=tkinter',
        '--clean',                        # Clean PyInstaller cache
        '--log-level=INFO',              # Detailed logging
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("Build completed successfully!")

if __name__ == "__main__":
    build_app()
