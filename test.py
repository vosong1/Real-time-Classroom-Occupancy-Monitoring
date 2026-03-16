import cv2
import numpy as np

img = cv2.imread("test.jpg")
if img is None:
    print("Không tìm thấy ảnh test.jpg")
    exit()
# Resize ảnh
resized = cv2.resize(img, (416, 416))

# Chuyển sang grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Sobel theo trục X
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Sobel theo trục Y
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Tính  magnitude của gradient
feature = cv2.magnitude(sobelx, sobely)

# Convert sang dạng ảnh
feature = cv2.convertScaleAbs(feature)

cv2.imshow("Original", img)
cv2.imshow("Resized", resized)
cv2.imshow("Gray", gray)
cv2.imshow("Edge Feature (Backbone Output)", feature)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("feature_output.jpg", feature)