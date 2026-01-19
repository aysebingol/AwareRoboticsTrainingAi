import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing(x): #trackbar oluştururken zorunlu fonk.
    pass

cv2.namedWindow("Trackbar")#trackbar ın görüneceği pencere

# HSV trackbarları #opencv de hue aralığı 0-180
cv2.createTrackbar("H Min", "Trackbar", 0, 180, nothing)
cv2.createTrackbar("H Max", "Trackbar", 180, 180, nothing)

#saturation(doygunluk) 0-255
cv2.createTrackbar("S Min", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbar", 255, 255, nothing)

#value(parlaklık) siyah nesneler için v değeri düşüktür
cv2.createTrackbar("V Min", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbar", 255, 255, nothing)

while True: #sonsuz döngü
    ret, frame = cap.read() #ret= görüntü alındı mı, frame= kameradan gelen görüntü
    if not ret: #görüntü gelmezse program durur
        break

    # HSV'ye çevir #kamera görüntüsü bgr formatındadır
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Trackbar değerlerini okuma #trackbarda seçilen değerleri alır,hue için
    h_min = cv2.getTrackbarPos("H Min", "Trackbar")
    h_max = cv2.getTrackbarPos("H Max", "Trackbar")

    #saturation trackbar kullanıcı seçimini okur
    s_min = cv2.getTrackbarPos("S Min", "Trackbar")
    s_max = cv2.getTrackbarPos("S Max", "Trackbar")

    #value trackbar kullanıcı seçimini okur
    v_min = cv2.getTrackbarPos("V Min", "Trackbar")
    v_max = cv2.getTrackbarPos("V Max", "Trackbar")

    #trackbardan gelen değerler, renk aralığı haline getirilir
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    #seçilen hsv aralığı beyaz, diğer yerler siyah
    mask = cv2.inRange(hsv, lower, upper)

    #sonuç görüntüsünü oluşturma, maskedeki beyaz alanlar korunur,diğer yerler silinir
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #görüntüleri ekranda gösterme
    cv2.imshow("Kamera", frame) #orijinal görüntü
    cv2.imshow("Maske", mask) #seçilen renkler
    cv2.imshow("Sonuc", result) #ayıklanmış görüntü

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#hsv? h-hue(renk tonu),s-saturation(doygunluk),v-value(parlaklık) - renk tespiti rgb ye göre daha kolaydır
#trackbar(kaydırma çubuğu), HSV, bir renk modelidir ve 3 bileşenden oluşur

#kamera açılır,görüntü hsv ye çevrilir,trackbar ile hsv değerleri değiştirilir,seçilen renkler ekranda gösterilir

