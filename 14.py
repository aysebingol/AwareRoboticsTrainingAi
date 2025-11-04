import cv2
import numpy as np


img = cv2.imread('yaprak.png')
if img is None:
    print("HATA: 'yaprak.png' bulunamadı.")
    exit()


cv2.namedWindow('HSV Trackbars')


def nothing(x):
    pass

#Her bir HSV değeri için trackbar ekleme
cv2.createTrackbar('H', 'HSV Trackbars', 0, 179, nothing)
cv2.createTrackbar('S', 'HSV Trackbars', 0, 255, nothing)
cv2.createTrackbar('V', 'HSV Trackbars', 0, 255, nothing)

while True:
    #HSV değerlerini okuma
    h = cv2.getTrackbarPos('H', 'HSV Trackbars')
    s = cv2.getTrackbarPos('S', 'HSV Trackbars')
    v = cv2.getTrackbarPos('V', 'HSV Trackbars')

    #Görüntüyü HSV formatına çevirme
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Tüm piksel değerlerine aynı HSV ayarını uygula
    h_new = (hsv[:,:,0] + h) % 180
    s_new = np.clip(hsv[:,:,1] + s, 0, 255)
    v_new = np.clip(hsv[:,:,2] + v, 0, 255)

   
    hsv_new = cv2.merge([h_new.astype(np.uint8), s_new.astype(np.uint8), v_new.astype(np.uint8)])


    bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV Trackbars', bgr_new)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
