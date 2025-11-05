import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
)
from PyQt5.QtCore import Qt
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class OCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Karakter Okuma Sistemi")
        self.setGeometry(300, 200, 600, 400)


        self.label = QLabel("Henüz bir dosya seçilmedi.", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)

   
        self.button = QPushButton("Resim Dosyası Seç ve Oku", self)
        self.button.clicked.connect(self.select_and_read)


        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_and_read(self):
        """Dosya seç ve OCR ile oku"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Bir Resim Dosyası Seç", "", "Görüntü Dosyaları (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            try:
                # OCR işlemi (İngilizce dil desteğiyle)
                text = pytesseract.image_to_string(Image.open(file_path))
                if not text.strip():
                    text = "Herhangi bir metin algılanamadı."
                self.label.setText(text)
            except Exception as e:
                self.label.setText(f"Hata oluştu: {e}")
        else:
            self.label.setText("Dosya seçilmedi.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())
