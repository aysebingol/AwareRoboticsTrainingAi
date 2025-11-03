import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageDraw
import numpy as np

class DummyDepthApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DepthEstimationV2 GUI - Dummy Test")
        self.setGeometry(100, 100, 1000, 600)

        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        image_layout = QHBoxLayout()

        # Butonlar
        self.btn_load_image = QPushButton("Resim Seç")
        self.btn_run = QPushButton("Derinlik Tahmini (Sahte)")
        button_layout.addWidget(self.btn_load_image)
        button_layout.addWidget(self.btn_run)

        # QLabel ile görüntüleme
        self.label_original = QLabel("Orijinal Resim")
        self.label_depth = QLabel("Sahte Derinlik Haritası")
        self.label_original.setFixedSize(480, 480)
        self.label_depth.setFixedSize(480, 480)
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_depth)

        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)
        self.setLayout(main_layout)

        # Fonksiyon bağlamaları
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_run.clicked.connect(self.run_dummy_depth)

        self.image_path = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path).scaled(self.label_original.width(), self.label_original.height())
            self.label_original.setPixmap(pixmap)

    def run_dummy_depth(self):
        if self.image_path is None:
            print("Önce resim seçin!")
            return

        # Orijinal resmi aç
        image = Image.open(self.image_path).convert("RGB")
        width, height = image.size

        # Sahte derinlik maskesi oluştur
        mask = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(mask)
        for y in range(height):
            intensity = int(255 * y / height)
            draw.line([(0, y), (width, y)], fill=(intensity, intensity, intensity))

        # QImage ve QLabel için dönüştürme
        mask_qimage = QImage(mask.tobytes(), mask.width, mask.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(mask_qimage).scaled(self.label_depth.width(), self.label_depth.height())
        self.label_depth.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DummyDepthApp()
    window.show()
    sys.exit(app.exec_())
