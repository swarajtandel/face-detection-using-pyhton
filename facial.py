import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Function to detect faces using OpenCV's Haar Cascade
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, gray

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face recognition model (you need to train this model beforehand)
if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    face_recognizer = cv2.createLBPHFaceRecognizer()

face_recognizer.read('trained_model.yml')

# Load images and corresponding labels for recognition
images = []
labels = []
students = {}
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if label not in students:
                students[label] = path
            images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            labels.append(int(label))

# Train the face recognition model
face_recognizer.train(images, np.array(labels))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Excel file
df = pd.DataFrame(columns=['Student', 'Arrival Time'])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces, _ = detect_faces(frame)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize face
        gray_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(gray_roi)
        if confidence < 50:  # You can adjust this threshold based on your needs
            student_name = os.path.basename(students[str(label)])
            if student_name not in df['Student'].values:
                df = df.append({'Student': student_name, 'Arrival Time': datetime.now()}, ignore_index=True)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the Excel file
df.to_excel('arrival_times.xlsx', index=False)
