import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('yaprak.png')

#edge detection (canny)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blurred, 100, 200)

#corner detection(harris)
gray_float = np.float32(gray)
dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
dst =cv2.dilate(dst, None)
img_with_corners = img.copy()
img_with_corners[dst > 0.01 * dst.max()] = [0, 0, 255]

#gorseli goster
plt.figure(figsize=(12,6))
plt.subplot(1,2,1), plt.imshow(edges, cmap='gray')
plt.title('edge detection (canny)')

plt.subplot(1,2,2), plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
plt.title('corner detection (harris)')
plt.axis('off')
plt.show()

#gorselleri kaydet
cv2.imwrite('edges.png', edges)
cv2.imwrite('corners.png', img_with_corners)
print("görseller kaydedildi: 'edges.png' ve ''corners.png")