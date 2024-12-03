@echo off
echo Installing required packages...
pip install pyinstaller
pip install -r requirements.txt

echo Building executable...
python build_app.py

echo Build complete! The executable is in the dist folder.
pause
