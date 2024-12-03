# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('c:\\Users\\NBP\\Desktop\\Internship\\SDK Development\\SDK Development\\Facial-Recognition-SDK\\models', 'models')]
datas += collect_data_files('dlib')
datas += collect_data_files('cv2')


a = Analysis(
    ['c:\\Users\\NBP\\Desktop\\Internship\\SDK Development\\SDK Development\\Facial-Recognition-SDK\\src\\facial_recognition_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['cv2', 'dlib', 'PIL', 'PIL.Image', 'PIL.ImageTk', 'numpy', 'mysql.connector', 'scipy', 'tkinter', 'tkinter.ttk', 'tkinter.messagebox'],
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
