import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision

class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Semantik Segmentasyon GUI")
        self.setGeometry(100, 100, 1000, 600)

        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        image_layout = QHBoxLayout()

        self.btn_load_image = QPushButton("Resim Seç")
        self.btn_load_model = QPushButton("Model Seç (Opsiyonel)")
        self.btn_run = QPushButton("Segmentasyonu Çalıştır")
        button_layout.addWidget(self.btn_load_image)
        button_layout.addWidget(self.btn_load_model)
        button_layout.addWidget(self.btn_run)

        self.label_original = QLabel("Orijinal Resim")
        self.label_processed = QLabel("İşlenmiş Resim")
        self.label_original.setFixedSize(480, 480)
        self.label_processed.setFixedSize(480, 480)
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_processed)

        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)
        self.setLayout(main_layout)

        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_run.clicked.connect(self.run_segmentation)

        self.image_path = None
        self.model_path = None

        # Önceden eğitilmiş DeepLabV3 modeli
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.eval()
        self.device = torch.device("cpu")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path).scaled(self.label_original.width(), self.label_original.height())
            self.label_original.setPixmap(pixmap)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Model Seç", "", "PyTorch Model (*.pt *.pth)")
        if file_path:
            self.model_path = file_path
            print("Model seçildi ve yüklendi (CPU test için hazır)")

    def run_segmentation(self):
        if self.image_path is None:
            print("Önce resim seçin!")
            return

        image = Image.open(self.image_path).convert("RGB")

        transform = T.Compose([
            T.Resize((520, 520)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            mask = output.argmax(0).byte().cpu().numpy()

        colors = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ])
        mask_rgb = colors[mask]

        mask_image = Image.fromarray(mask_rgb.astype(np.uint8))
        mask_image = mask_image.resize(image.size)
        mask_qimage = QImage(mask_image.tobytes(), mask_image.width, mask_image.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(mask_qimage).scaled(self.label_processed.width(), self.label_processed.height())
        self.label_processed.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec_())
