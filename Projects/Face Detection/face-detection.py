import cv2
import sys
import os

# Get the camera object
cam = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)

# Get the script directory for the cascade file
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")

# Initialize the haar cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if cascade loaded successfully
if face_cascade.empty():
    print("Error: Could not load face cascade classifier.")
    print(f"Path checked: {cascade_path}")
    sys.exit(1)

# Main Loop
try:
    while True:
        # Read the frame from the camera
        ret, frame = cam.read()

        # Check if frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame into grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all the faces in the grayscale frame
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

        # Draw a blue rectangle for each face in the frame
        for x, y, width, height in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

        # Add face count overlay
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Quit button
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    # Release resources
    cam.release()
    cv2.destroyAllWindows()
