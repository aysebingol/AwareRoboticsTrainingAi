import cv2
import numpy as np
from matplotlib import pyplot as plt

# Orijinal görseli oku
img = cv2.imread('lacoste.png')

# Griye dönüştür
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Griyi 3 kanala genişlet
gray_stack = cv2.merge([gray, gray, gray])

# Farkı hesapla
diff = img.astype(int) - gray_stack.astype(int)

# Farkın mutlak değerini al ve normalize et (0-255 arası)
diff_abs = np.abs(diff).astype(np.uint8)
diff_norm = cv2.normalize(diff_abs, None, 0, 255, cv2.NORM_MINMAX)

# Görsel olarak göster
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Orijinal')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(gray, cmap='gray')
plt.title('Gri')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(diff_norm)
plt.title('Fark')
plt.axis('off')

plt.show()

cv2.imwrite('diff_visual.png', diff_norm)
np.save('diff_array.npy', diff)
