# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['facial_recognition_app.py'],
    pathex=['.'],  # Current working directory
    binaries=[],
    datas=[
        ('C:/Users/NBP/Desktop/New/Facial-Recognition-SDK/models/models/shape_predictor_68_face_landmarks.dat', 'models'),
        ('C:/Users/NBP/Desktop/New/Facial-Recognition-SDK/models/models/dlib_face_recognition_resnet_model_v1.dat', 'models'),
        ('C:/Users/NBP/Desktop/New/Facial-Recognition-SDK/models/models/temp', 'temp'),  # Include temp directory
    ],
    hiddenimports=[
        'mysql.connector.locales',
        'mysql.connector.locales.eng',
        'mysql.connector.plugins.mysql_native_password',
        'PIL._tkinter_finder',
        'cv2',
        'dlib',
        'numpy',
        'tkinter',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceRecognitionApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI Only
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='FaceRecognitionApp',
)
