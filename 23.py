import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QHBoxLayout, QVBoxLayout, QFileDialog, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

class SegmentasyonArayuzu(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Aware Robotics - Semantik Segmentasyon Analiz Paneli')
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet("background-color: #2c3e50; color: white;")

        main_layout = QVBoxLayout()

        baslik = QLabel("YOLO SEMANTİK SEGMENTASYON ANALİZİ")
        baslik.setFont(QFont('Arial', 18, QFont.Bold))
        baslik.setAlignment(Qt.AlignCenter)
        baslik.setStyleSheet("margin: 10px; color: #ecf0f1;")
        main_layout.addWidget(baslik)

        button_layout = QHBoxLayout()
        
        self.btn_model = QPushButton('📁 1. Modeli Seç (.pt)')
        self.btn_resim = QPushButton('🖼️ 2. Resim Seç ve Analiz Et')
        
        stil = """
            QPushButton {
                background-color: #34495e; border: 2px solid #3498db;
                border-radius: 10px; padding: 10px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3498db; }
        """
        self.btn_model.setStyleSheet(stil)
        self.btn_resim.setStyleSheet(stil.replace("#3498db", "#2ecc71"))
        
        self.btn_model.clicked.connect(self.model_sec)
        self.btn_resim.clicked.connect(self.analiz_yap)
        
        button_layout.addWidget(self.btn_model)
        button_layout.addWidget(self.btn_resim)
        main_layout.addLayout(button_layout)

        display_layout = QHBoxLayout()
        
        self.lbl_sol = self.cerceve_olustur("Orijinal Resim Bekleniyor...")
        self.lbl_sag = self.cerceve_olustur("Segmentasyon Sonucu...")
        
        display_layout.addWidget(self.lbl_sol)
        display_layout.addWidget(self.lbl_sag)
        main_layout.addLayout(display_layout)

        self.lbl_durum = QLabel("Sistem Hazır. Lütfen önce modeli yükleyin.")
        self.lbl_durum.setAlignment(Qt.AlignCenter)
        self.lbl_durum.setStyleSheet("color: #bdc3c7; font-size: 12px; margin-top: 10px;")
        main_layout.addWidget(self.lbl_durum)

        self.setLayout(main_layout)

    def cerceve_olustur(self, metin):
        lbl = QLabel(metin)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFrameShape(QFrame.Box)
        lbl.setLineWidth(2)
        lbl.setMinimumSize(500, 500)
        lbl.setStyleSheet("background-color: #34495e; border-radius: 5px; color: #95a5a6;")
        return lbl

    def model_sec(self):
        yol, _ = QFileDialog.getOpenFileName(self, "Model Dosyası Seç", "", "Modeller (*.pt)")
        if yol:
            try:
                self.lbl_durum.setText("Model yükleniyor...")
                QApplication.processEvents()
                self.model = YOLO(yol).to('cpu')
                self.btn_model.setText(f"Model: {os.path.basename(yol)}")
                self.lbl_durum.setText("✅ Model yüklendi. Şimdi resim seçebilirsiniz.")
            except Exception as e:
                self.lbl_durum.setText(f"❌ Model Hatası: {e}")

    def analiz_yap(self):
        if self.model is None:
            self.lbl_durum.setText("⚠️ Lütfen önce modeli yükleyin!")
            return

        yol, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resimler (*.jpg *.png *.jpeg)")
        if yol:
            pix = QPixmap(yol)
            self.lbl_sol.setPixmap(pix.scaled(500, 500, Qt.KeepAspectRatio))
            
            self.lbl_durum.setText("Analiz ediliyor...")
            QApplication.processEvents()

            try:
                img = cv2.imread(yol)
                results = self.model.predict(source=img, conf=0.25, device='cpu')
                sonuc_img = results[0].plot() # Maskeleri çizer

                rgb = cv2.cvtColor(sonuc_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                
                self.lbl_sag.setPixmap(QPixmap.fromImage(q_img).scaled(500, 500, Qt.KeepAspectRatio))
                self.lbl_durum.setText("✅ Analiz başarıyla tamamlandı!")
            except Exception as e:
                self.lbl_durum.setText(f"❌ Analiz Hatası: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SegmentasyonArayuzu()
    ex.show()
    sys.exit(app.exec_())