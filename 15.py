import cv2
import numpy as np


cap = cv2.VideoCapture('/home/aysebingol/Downloads/BlueBallAdventure.mp4')#video dosyasını açma

if not cap.isOpened(): #dosya bulunamazsa proje boşa çalışmasın
    print("HATA: Video dosyası açılamadı.")
    exit()

while True: #sonsuz döngü
    ret, frame = cap.read() #ret= görüntü alındı mı,kare geldi mi?(true/false), frame(o anki görüntü)= kameradan gelen görüntü
    if not ret: #video biterse başa sarılır,baştan oynatılır
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#bgr den hsv dönüşümü

    #bu değerler hsv formatında kırmızı rengi verir,program sadece bu aralıktaki pikselleri seçer
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])


    #kırmızı renge uyan pikseller beyaz,diğer tüm pikseller siyah,renk tespiti yapılan asıl adımdır
    mask = cv2.inRange(hsv, lower_red, upper_red)

    #tespit edilen rengi görüntüleme,sadece kırmızı alanlar görüntüde kalır diğer bölgeler gizlenir
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Orijinal', frame) #orijinal video
    cv2.imshow('Kırmızı Renk Tespiti', result) #sadece kırmızı renkler

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
