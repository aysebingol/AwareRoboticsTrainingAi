# Gerekli Kütüphaneler
import sys 
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 


try:
    midas_model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type) 
    print(f"MiDaS modeli ({midas_model_type}) yüklendi.")
except Exception as e:
    print(f"Hata: Model yüklenemedi. Tahmin devre dışı. Hata: {e}")
    midas = None 

class DepthEstimationApp(QWidget):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Derinlik Tahmini (MiDaS) Arayüzü")
        self.setGeometry(100, 100, 1000, 600)

        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        image_layout = QHBoxLayout()

        self.btn_load_image = QPushButton("1. Resmi Seç")
        self.btn_run = QPushButton("2. Derinlik Tahminini Çalıştır")
        
        button_layout.addWidget(self.btn_load_image)
        button_layout.addWidget(self.btn_run)

        self.label_original = QLabel("Orijinal Resim (Seçilmeli)")
        self.label_depth = QLabel("Derinlik Haritası (Siyah=Yakın, Beyaz=Uzak)")
        
        self.label_original.setFixedSize(480, 480)
        self.label_depth.setFixedSize(480, 480)
        
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_depth)

        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)
        self.setLayout(main_layout)

        # Fonksiyon bağlantıları
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_run.clicked.connect(self.run_depth_estimation) 

        self.image_path = None
        self.model = midas
        
        if self.model:
            self.model.eval() 
            self.device = torch.device("cpu") 
            print("Model hazır: Gerçek Derinlik Tahmini çalışmaya hazır.")
        else:
             self.device = None
             self.btn_run.setEnabled(False) 
             print("Model yüklenemediği için tahmin butonu devre dışı.")


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")
        
        if file_path:
            self.image_path = file_path
            
            pixmap = QPixmap(file_path).scaled(self.label_original.width(), self.label_original.height())
            self.label_original.setPixmap(pixmap)
            print(f"Resim yüklendi: {file_path}")

    def run_depth_estimation(self):
        if self.image_path is None or self.model is None:
            print("Hata: Lütfen resim seçin veya model yüklenemedi.")
            return

        image = Image.open(self.image_path).convert("RGB")
        
        transform = T.Compose([
            T.Resize(384, interpolation=InterpolationMode.BICUBIC), 
            T.CenterCrop(384), 
            T.ToTensor(),  
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # Tahmini çalıştır
        with torch.no_grad(): 
            prediction = self.model(input_tensor)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1], 
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Görselleştirme için normalize etme (0-255 aralığına)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255).astype(np.uint8)
        
        depth_image = Image.fromarray(depth_map, mode='L') 
        
        # Sonucu PyQtde göster
        qimage = QImage(depth_image.tobytes(), depth_image.width, depth_image.height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(self.label_depth.width(), self.label_depth.height())
        
        self.label_depth.setPixmap(pixmap)
        print("Derinlik tahmini tamamlandı.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DepthEstimationApp()
    window.show()
    sys.exit(app.exec_())
