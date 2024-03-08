import cv2

# Path to the cascade classifier file
cascade_file_path = "C:\\xampp\\htdocs\\object-detection\\kk\\Face-Detection-main\\haarcascade_frontalface_default.xml"

# Load the cascade classifier
cascade = cv2.CascadeClassifier(cascade_file_path)

# Check if the cascade classifier is loaded successfully
if cascade.empty():
    print("Error: Unable to load cascade classifier file:", cascade_file_path)
    exit()

# Initialize camera
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = cascade.detectMultiScale(gray_img)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Face Detection", img)
    
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
