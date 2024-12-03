@echo off
echo Testing FaceRecognition.exe...
echo.

REM Check if models directory exists in dist folder
if not exist "dist\models" (
    echo Copying models directory...
    xcopy /E /I /Y "models" "dist\models"
)

echo Running FaceRecognition.exe...
cd dist
FaceRecognition.exe
cd ..

echo.
if errorlevel 1 (
    echo Error: Application exited with code %errorlevel%
) else (
    echo Application closed successfully
)
pause
