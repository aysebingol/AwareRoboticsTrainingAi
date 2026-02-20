import sys
import os
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = "xcb"

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        
        self.setWindowTitle("Aware Robotics - Kamera Takip Sistemi")
        self.resize(800, 650)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.label_camera = QLabel("Kamera Bekleniyor...")
        self.label_camera.setAlignment(Qt.AlignCenter)
        self.label_camera.setStyleSheet("background-color: black; border: 2px solid #27ae60; color: white;")
        self.label_camera.setMinimumSize(640, 480)
        
        self.button_layout = QHBoxLayout()
        self.button_baslat = QPushButton("▶ Kamerayı Başlat")
        self.button_durdur = QPushButton("⏹ Kamerayı Durdur")
        
        btn_style = "height: 40px; font-weight: bold; border-radius: 5px;"
        self.button_baslat.setStyleSheet(btn_style + "background-color: #2ecc71; color: white;")
        self.button_durdur.setStyleSheet(btn_style + "background-color: #e74c3c; color: white;")
        
        self.button_layout.addWidget(self.button_baslat)
        self.button_layout.addWidget(self.button_durdur)
        
        self.main_layout.addWidget(self.label_camera)
        self.main_layout.addLayout(self.button_layout)

        self.model = YOLO("yolov8n.pt")
        self.model.to('cpu') 

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.button_baslat.clicked.connect(self.start_camera)
        self.button_durdur.clicked.connect(self.stop_camera)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            self.label_camera.setText("HATA: Kamera Açılamadı!")

    def stop_camera(self):
        self.timer.stop() 
        if hasattr(self, 'cap'):
            self.cap.release()
        self.label_camera.clear()
        self.label_camera.setText("Kamera Durduruldu.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, device='cpu', verbose=False)
            
            kutulanmis = results[0].plot()
            rgb_img = cv2.cvtColor(kutulanmis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w

            convert_Qt_format = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_Qt_format.scaled(self.label_camera.width(), self.label_camera.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.label_camera.setPixmap(QPixmap.fromImage(p))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())