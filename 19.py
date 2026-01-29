import cv2
import numpy as np
import time

#yolo model dosya yolları
weights_path = "/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/yolov4/yolov4.weights"
config_path = "/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/yolov4/yolov4.cfg"

#yolov4 modelini yükle(readNetFromDarknet)
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

cap = cv2.VideoCapture(0) #1 harivi kamera,0 bilgisayarın kamerası

CONF_THRESHOLD = 0.5 #%50den emin değilse gösterme
NMS_THRESHOLD = 0.4 #çakışan kutuları temizle

prev_time = 0 #fps saniyede kaç kare işleniyor,hesaplamak için önceki zamanı saklıyoruz


while True: #kameradan süreki görüntüyü al
    ret, frame = cap.read() #ret-görüntü geldimi,frame-kameradan gelen görüntü
    if not ret:
        break

    H, W = frame.shape[:2]#görüntü boyutunu alıyoruz

    #görüntüyü yolo formatına çeviriyoruz
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)

    #görüntüyü modele çeviriyoruz
    net.setInput(blob)

    #yolonun nesne tahmini yaptığı son kaktmanları seçiyoruz
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    #modeli çalıştırıyoruz,yolo görüntüyü analiz eder,insan var mı diye tahmin eder
    outputs = net.forward(output_layers)

    boxes = [] #kutunun koordinatları
    confidences = [] #güven skorları

    
    for output in outputs:
        for detection in output:
            #en güçlü sınıfı buluyoruz
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            #sadece insanları seçiyoruz,cocoda 0 insan
            if confidence > CONF_THRESHOLD and class_id == 0:
                #kutu koordinatlarını hesaplama
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")

                #yolo merkezden ölçer,opencv sol üst köşeden ölçer bu yüzden çeviriyoruz
                #sol üst köşeyi buluyoruz
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                #kutuları kaydediyoruz
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))

    #çakışan kutuları temizleme(nms)-aynı kişi için 3 kutu varsa sadece en iyisini bırakır
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            #kutuları ekrana çiziyoruz,insan etrafına yeşil kutu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #güven skoru yazıyoruz,modelin ne kadar emin olduğunu gösteririrz
            label = f"Person: {confidences[i]:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #fps hesaplama=saniyede işlenen kare sayısı
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    #fps değerini sağ üst köşeye yazıyoruz
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (W - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("YOLOv4 Person Detection + FPS", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
