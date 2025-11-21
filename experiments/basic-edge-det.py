import cv2
import numpy as np

image = cv2.imread("D:\\Competition\\NASARoverRally\\RUET-Exover25\\assets\\extracted_images\\frame_01500.jpg")

# Controling Brightness and Contrast
alpha = 2.0
beta = 32

adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Convert to GrayScale Image
gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

kernel = 5

# Converting Gray Image to Blur Image
blurred_image = cv2.GaussianBlur(gray_image, (kernel, kernel), 0)

# Applying Canny Edge
edges = cv2.Canny(blurred_image, 100, 200)

cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()