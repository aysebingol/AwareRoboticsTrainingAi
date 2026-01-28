import cv2
import numpy as np

#yolov4 model dosyaları cfg=modelin mimarisi - weights=modelin öğrendiği bilgiler
weights_path = "/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/yolov4/yolov4.weights"
config_path = "/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/yolov4/yolov4.cfg"

#yolo modelini opencv ile yükleme(readNetFromDarknet),yolov4 beynini programa takıyoruz
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

cap = cv2.VideoCapture(0)

CONF_THRESHOLD = 0.3  #güven eşiği,0.9=çok emin,0.2=pek emin değil

while True: #görüntü boyutunu alma
    ret, frame = cap.read()
    if not ret:
        break

    (H, W) = frame.shape[:2]  # Kare boyutları

    #blob=görüntüyü modelin anlayacağı formata çevirme
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)#görüntüyü modele verme

    #çıkış katmanları alınır, yolonun nesne tahmini yaptığı son katmanlar
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    #modeli çalıştırma
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    # Model çıktısını analiz et
    for output in outputs:
        for detection in output:
            #sadece insanları seçme
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            #sadece insan (class_id == 0)
            if confidence > CONF_THRESHOLD and class_id == 0:
                #kutu koordinatlarını hesaplama
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")

                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            #algılanan insanın çevresine yeşil kutu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #confidence ekrana yazdırma, modelin ne kadar emin olduğunu gösterir
            label = f"Person: {confidences[i]:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv4 Insan Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#confidence değerini düşürünce 0.3(Emin olmasa bile tahmin yap) daha çok insan algılar,yanlış tespit artar
#0.5(Dengeli) dengeli,önerilen
#0.7(Çok emin değilse gösterme) daha az insan,ama daha doğru
#confidence değeri artırıldığında yanlış tespitler azalmaktadır ancak bazı insanlar algılanamayabilmektedir.confidence azaltıldığında ise daha fazla insan algılanmakta fakat yanlış tespit oranı artmaktadır.

