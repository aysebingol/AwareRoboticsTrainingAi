import cv2
import numpy as np

# --- 1. Model Yollarını Tanımla ---
weights_path = "yolov4-tiny-3l_best.weights"
cfg_path = "yolov4-tiny-3l.cfg"
names_path = "coco.names"
input_path = "pers.mp4"
output_video_path = "output_video.avi"

# --- 2. Sınıf İsimlerini Yükle ---
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("Sınıf isimleri yüklendi:", classes)


net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("YOLOv4 modeli başarıyla yüklendi.")


# --- 4. Tespit Fonksiyonu ---
def detect_objects(frame, confidence_threshold, nms_threshold=0.4):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Güven eşiğini aşan tüm tespitleri al
            if confidence > confidence_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    results = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            results.append({
                "box": boxes[i],
                "confidence": confidences[i],
                "class_id": class_ids[i]
            })
    return results


# --- 5. Tespitleri Görselleştir ---
def draw_detections(frame, detections):
    for det in detections:
        x, y, w, h = det["box"]
        confidence = det["confidence"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{confidence:.2f}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


# --- 6. Video İşleme Fonksiyonu ---
def process_video(video_path, confidence_thresholds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: Video açılamadı: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_name = "output_video.avi"
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    print(f"Video işleniyor: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = detect_objects(frame, 0.5)
        drawn_frame = draw_detections(frame, detections)
        out.write(drawn_frame)

        # Ekranda göster (test için)
        cv2.imshow("Sonuç Videosu", drawn_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 30 == 0:
            print(f"{frame_count} kare işlendi...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video işleme tamamlandı, kaydedildi:", output_name)


# --- 7. Ana Program ---
if __name__ == "__main__":
    if input_path.endswith(('.mp4', '.avi', '.mov', '.webm')):
        process_video(input_path, [0.5])
    else:
        print("Desteklenmeyen dosya formatı.")

print("Uygulama tamamlandı.")
