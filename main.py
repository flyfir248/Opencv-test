import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the image
img = cv2.imread("image.jpg")

# Convert into grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

# Draw rectangle around the faces
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)

# Display the output
cv2.imshow("Faces", img)
cv2.waitKey()
cv2.destroyAllWindows()