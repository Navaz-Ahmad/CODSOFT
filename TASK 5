import cv2
import face_recognition
import numpy as np

# Step 1: Initialize the face detector
# We are using Haar cascades for face detection, which is pre-trained on frontal face data
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Load a sample image and encode the face for recognition
# In a real application, you'd load multiple known faces
known_image = face_recognition.load_image_file("known_person.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]  # Encode the known face

# Store the known face and its corresponding name
known_face_encodings = [known_face_encoding]
known_face_names = ["Known Person"]

# Step 3: Define the function for face detection and recognition
def detect_and_recognize_faces(frame):
    # Convert the frame to grayscale for face detection using Haar cascades
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If no faces are detected, we return early
    if len(faces) == 0:
        return frame
    
    # Convert the frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Step 4: Recognize faces using deep learning-based face_recognition library
    # Get the locations and encodings of all faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Step 5: Loop over each detected face and perform recognition
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # If a match was found, we retrieve the name of the matched face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Step 6: Draw bounding boxes around detected faces and label them
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw the label with the name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    return frame

# Step 7: Real-time face detection and recognition via webcam
def run_face_detection_and_recognition():
    # Open a connection to the webcam (0 is the default camera)
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Perform face detection and recognition on the frame
        output_frame = detect_and_recognize_faces(frame)
        
        # Display the output frame with face detection and recognition
        cv2.imshow('Face Detection and Recognition', output_frame)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the face detection and recognition system
run_face_detection_and_recognition()
