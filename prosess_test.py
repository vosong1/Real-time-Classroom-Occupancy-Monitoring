import cv2
import numpy as np
IMAGES = ["test.jpg", "classroom.jpg"]

for img_path in IMAGES:

    print("Processing:", img_path)

    img = cv2.imread(img_path)

    if img is None:
        print("Cannot load image:", img_path)
        continue
    resized = cv2.resize(img, (416, 416))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    feature = cv2.magnitude(sobelx, sobely)
    feature = cv2.convertScaleAbs(feature)

    cv2.imshow("Original Image", img)
    cv2.imshow("Resized Image", resized)
    cv2.imshow("Feature Map (Edge)", feature)

    cv2.waitKey(0)

cv2.destroyAllWindows()