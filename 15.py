import cv2
import numpy as np


cap = cv2.VideoCapture('test.mp4')

if not cap.isOpened():
    print("HATA: Video dosyası açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Orijinal', frame)
    cv2.imshow('Kırmızı Renk Tespiti', result)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
