import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('kinder.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#kernel oluştur
kernel = np.ones((5,5), np.uint8)


#erosion
erosion = cv2.erode(binary, kernel, iterations=1)

#dilation
dilation = cv2.dilate(binary, kernel, iterations=1)

#opening
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

#gorselleri goster
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(binary, cmap='gray')
plt.title('Orijinal Binary')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.show()

# Görselleri kaydet
cv2.imwrite('binary.png', binary)
cv2.imwrite('erosion.png', erosion)
cv2.imwrite('dilation.png', dilation)
cv2.imwrite('opening.png', opening)

print("Görseller kaydedildi: binary.png, erosion.png, dilation.png, opening.png")
