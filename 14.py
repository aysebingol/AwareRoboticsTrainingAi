import cv2
import numpy as np

cap = cv2.VideoCapture('test.mp4')

if not cap.isOpened():
    print("HATA: Video dosyası açılamadı.")
    exit()

cv2.namedWindow('HSV Trackbars')

def nothing(x):
    pass


cv2.createTrackbar('H', 'HSV Trackbars', 0, 179, nothing)
cv2.createTrackbar('S', 'HSV Trackbars', 0, 255, nothing)
cv2.createTrackbar('V', 'HSV Trackbars', 0, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti")
        break


    h = cv2.getTrackbarPos('H', 'HSV Trackbars')
    s = cv2.getTrackbarPos('S', 'HSV Trackbars')
    v = cv2.getTrackbarPos('V', 'HSV Trackbars')


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    h_new = (hsv[:,:,0] + h) % 180
    s_new = np.clip(hsv[:,:,1] + s, 0, 255)
    v_new = np.clip(hsv[:,:,2] + v, 0, 255)


    hsv_new = cv2.merge([h_new.astype(np.uint8), s_new.astype(np.uint8), v_new.astype(np.uint8)])
    bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


    cv2.imshow('HSV Trackbars', bgr_new)


    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
