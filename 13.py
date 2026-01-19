import cv2 #kamera + görüntü işleme
import numpy as np #sayılar ve renk aralıklarını tanımlamak için

cap = cv2.VideoCapture(0) #kamerayı açıyoruz, 0 varsayılan kamera

while True: #sonsuz döngü, kamera biz kapatana kadar açık kalır
    ret, frame = cap.read() #kameradan götüntü alma( ret= true/false),frame=kameradan gelen anlık görüntü
    if not ret: #eğer görüntü gelmezse programdan çıkılır
        break

    #görüntü hsv ye çevrilir,kamera görüntüsü bgr formatındadır,hsv renk ayırmayı daha kolay yapar, siyah rengi hsv den ayırmak daha kolay
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #siyah renk aralığını belirleme, bu aralık koyu(parlaklığı düşük) alanları seçer,yani siyah koyu gri
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    #siyah yerler beyaz, diğer yerler siyah olur, bilgisayar artık siyah alanlara bakıyor
    mask = cv2.inRange(hsv, lower_black, upper_black)

    #siyah nesnelerin dış sınırlarını bulur,beyaz alan nerede başlıyor nerede bitiyor
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #bulunan nesneleri tek tek inceliyoruz
    for cnt in contours:
        #nesnenin alanını ölçme
        area = cv2.contourArea(cnt)

        if area > 1000:
            #nesnelerin etrafına kutu çizilir
            x, y, w, h = cv2.boundingRect(cnt)#nesneleri bulur v2.boundingRect
            #kutunun üstüne yazı yazar, bu bir siyah nesne demek için
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#yeşil bir diktörtgen çizer, 2 çizgi kalınlığı
            cv2.putText(frame, "Siyah Nesne", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Kamera", frame) #normal görüntü
    cv2.imshow("Maske", mask) #siyah alanalrın beyaz hali

    #q tuşuna basınca program durur
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#kamerayı serbest bırakır, açık pencereleri kapatır
cap.release()
cv2.destroyAllWindows()

#kamera açılır,siyah renkler ayrılır,siyah nesneler bulunur,etraflarına kutu çizilir
