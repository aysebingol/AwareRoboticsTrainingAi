import sys
import os
import cv2
import pytesseract
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

os.environ["QT_QPA_PLATFORM"] = "xcb"

class OCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Aware Robotics - Akıllı OCR Okuma Sistemi")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: #f5f6fa;")

        main_layout = QVBoxLayout()

        self.btn_sec = QPushButton("Analiz Edilecek Resmi Seç")
        self.btn_sec.setFixedHeight(50)
        self.btn_sec.setStyleSheet("""
            QPushButton { background-color: #2f3640; color: white; font-weight: bold; border-radius: 8px; }
            QPushButton:hover { background-color: #718093; }
        """)
        self.btn_sec.clicked.connect(self.resim_analiz_et)
        main_layout.addWidget(self.btn_sec)

        content_layout = QHBoxLayout()

        self.lbl_image = QLabel("Resim Burada Görünecek")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("border: 2px dashed #dcdde1; background-color: white; border-radius: 10px;")
        self.lbl_image.setMinimumSize(450, 450)
        content_layout.addWidget(self.lbl_image)

        self.txt_sonuc = QTextEdit()
        self.txt_sonuc.setPlaceholderText("Okunan metinler burada görünecek...")
        self.txt_sonuc.setReadOnly(True)
        self.txt_sonuc.setStyleSheet("""
            QTextEdit { 
                border: 2px solid #2f3640; 
                background-color: #ffffff; 
                border-radius: 10px; 
                padding: 10px; 
                font-size: 14px;
                color: #2f3640;
            }
        """)
        content_layout.addWidget(self.txt_sonuc)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

    def resim_analiz_et(self):
        yol, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resimler (*.png *.jpg *.jpeg)")
        
        if yol:
            pixmap = QPixmap(yol)
            self.lbl_image.setPixmap(pixmap.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            img = cv2.imread(yol)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            
            try:
                text = pytesseract.image_to_string(gray, lang='tur+eng')
                
                if text.strip():
                    self.txt_sonuc.setText(text)
                else:
                    self.txt_sonuc.setText("Resimde okunabilir bir metin bulunamadı.")
            except Exception as e:
                self.txt_sonuc.setText(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())