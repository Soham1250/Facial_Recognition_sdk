import cv2

# Open the video stream
cap = cv2.VideoCapture(0)

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Load the pre-trained Haar Cascade classifier file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables
face_detected = False

def detect_faces(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

# while not face_detected:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     print("H")
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     if len(faces) > 0:
#         # Stop capturing the live feed
#         face_detected = True

#         # Draw rectangles around detected faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Display the frame with the highlighted face

#         cv2.imshow('Detected Face', frame)
        
#         # Wait for a key press and then break
#         cv2.waitKey(500)

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
