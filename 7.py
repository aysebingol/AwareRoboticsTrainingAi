import cv2
import numpy as np


img = cv2.imread('yaprak.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#hhreshold işlemi
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


#gürültü temizliği
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#arka plan ve ön plan belirleme
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unkown = cv2.subtract(sure_bg, sure_fg)

#marker oluşturma
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unkown == 255] = 0

#watershed algoritması
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

#sonuç
cv2.imwrite("watershed_sonuc.png", img)
cv2.imshow('watershed segmentasyon', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
