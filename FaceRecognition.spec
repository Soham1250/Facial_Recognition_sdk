# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\facial_recognition_app.py'],
    pathex=[],
    binaries=[],
    datas=[('c:\\Users\\NBP\\Desktop\\Internship\\SDK Development\\SDK Development\\Facial-Recognition-SDK\\models\\shape_predictor_68_face_landmarks.dat', 'models'), ('c:\\Users\\NBP\\Desktop\\Internship\\SDK Development\\SDK Development\\Facial-Recognition-SDK\\models\\dlib_face_recognition_resnet_model_v1.dat', 'models')],
    hiddenimports=['mysql.connector.locales', 'mysql.connector.locales.eng', 'mysql.connector.plugins', 'mysql.connector.plugins.mysql_native_password', 'PIL._tkinter_finder', 'dlib', 'cv2', 'numpy', 'tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FaceRecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
