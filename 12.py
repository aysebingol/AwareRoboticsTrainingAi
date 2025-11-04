import cv2

video_path = "test.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okunamadı.")
        break

    cv2.imshow("Kamera Simülasyonu", frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
