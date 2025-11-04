import cv2
import numpy as np

weights_path = "yolov3-wider_16000.weights"
config_path = "yolov3-face.cfg"

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#Video veya resim kaynağı(kamera açılmıyor)
video_path = "person.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []


    for output in layer_outputs:
        for detection in output:
            confidence = detection[4]
            if confidence > 0.5:  
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    if len(indexes) > 0:
        for i in indexes.flatten():
            (x, y, w, h) = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Yuz", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("YOLOv3 Yuz Tespiti", frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
