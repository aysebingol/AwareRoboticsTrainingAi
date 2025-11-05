import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# Haar Cascade dosyası (OpenCV'nin kendi içinde var)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.active = False
        self.cam = None
        self.face_ids = {}
        self.next_id = 1

    def run(self):
        self.cam = cv2.VideoCapture(0)
        self.active = True

        while self.active:
            ret, frame = self.cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                cx, cy = x + w // 2, y + h // 2
                face_id = None

                # Daha önceki yüzlerle mesafeyi kontrol et
                for fid, (px, py) in self.face_ids.items():
                    dist = ((cx - px)**2 + (cy - py)**2) ** 0.5
                    if dist < 100:
                        face_id = fid
                        self.face_ids[fid] = (cx, cy)
                        break

                if face_id is None:
                    face_id = self.next_id
                    self.face_ids[face_id] = (cx, cy)
                    self.next_id += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 150, 255), 2)
                cv2.putText(frame, f"ID: {face_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

            # Görüntüyü Qt formatına çevir
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flipped = cv2.flip(rgb, 1)
            qt_image = QImage(flipped.data, flipped.shape[1], flipped.shape[0], QImage.Format_RGB888)
            self.frame_ready.emit(qt_image)

    def stop(self):
        self.active = False
        if self.cam:
            self.cam.release()
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kamera Uygulaması")
        self.setGeometry(200, 100, 800, 600)

        self.label = QLabel("Kamera kapalı")
        self.label.setAlignment(Qt.AlignCenter)

        self.start_btn = QPushButton("Başlat")
        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = None

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

    def start_camera(self):
        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.label.clear()
        self.label.setText("Kamera kapalı")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()



