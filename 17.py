import cv2
import numpy as np

#cfg yolov3 mimarisini tanımlar(katmanlar,filtreler), weights önceden eğitilmiş ağırlıklar(modelin öğrendiği yüz bilgileri)
weights_path = "/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/yolov3-wider_16000.weights"
config_path = "/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/yolov3-face.cfg"

net = cv2.dnn.readNetFromDarknet(#opencv ile yolov3 medelini yükler
    config_path, weights_path)

#video dosyası açılır
video_path = "/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/pers.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened(): #isopened video açılamadıysa programı durdurur
    print("Video açılamadı.")
    exit()

while True:
    ret, frame = cap.read() #videodan bir kare alınır, frame(alınan kare görüntüsü)
    if not ret: #true(kare alındı)/false(video biter) kare alınıp alınmadığı kontrol edilir
        break

    (H, W) = frame.shape[:2] #karenin boyutları
    #blob(modelin anlayacağı görüntü formatı) modelin anlayacağı format
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), #1 / 255.0,=normalizasyon,(416, 416)=yolo giriş boyutu
                                 swapRB=True, crop=False) #crop=false kırpma yok
    net.setInput(blob) #görüntüyü modele verme(kareyi inceler)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()
                     .flatten()]

    #layer_outputs=her yüz için koordinatlar+güven skorları
    layer_outputs = net.forward(output_layers)

    boxes = [] #yüz koordinatları
    confidences = [] #ne kadar eminim?

#her detection bunları içerir= [x, y, w, h, confidence]
    for output in layer_outputs:
        for detection in output:
            confidence = detection[4] #güven eşiği %50 den azsa emin değilim kutuya alma
            if confidence > 0.5:  
                box = detection[0:4] * np.array([W, H, W, H]) #kutuyu piksele çevirme
                (centerX, centerY, width, height) = box.astype("int")
                
                #sol üst köşeyi bulma(opencv diktörgeni sol üst köşeden çizer)
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    #aynı yüz için birden fazla kutu olabilir, nms sadece en güvenli kutuyu bırakır, 0.5 güven skoru eşik değeri, 0.4 kutuların çakışma toleransı
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    if len(indexes) > 0:
        for i in indexes.flatten():
            (x, y, w, h) = boxes[i]
            cv2.rectangle( #kutu çizme
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.putText( #yüz etrafına yeşil kutu
                frame, "Yuz", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("yolov3 yüz tespiti", frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


#video kaynağını kapat, açık tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()


#YOLO (You Only Look Once)=gerçek zamanlı nesne algılama algoritması,bu algoritma CNN yöntemini kullanır=bu yöntem ile görüntüyü bölgelere ayırır ve her bölge için olasıkları tahmin eder bu sayede doğruluk oranı yüksektir
#cfg=konfigürasyon dosyası-weights=model-nesne isim dosyası=coco.names(eğitilen nesnelerin adlarını içerir)
#cfg modelin mimarisi-weights modelin öğrendiği bilgiler