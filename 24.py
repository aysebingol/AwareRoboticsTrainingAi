import os
import sys
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QHBoxLayout, QVBoxLayout, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

os.environ["QT_QPA_PLATFORM"] = "xcb"

class DepthEstimationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu") 
        self.model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform if self.model_type == "MiDaS_small" else midas_transforms.dpt_transform

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Aware Robotics - DepthEstimationV2 Analiz Paneli')
        self.setGeometry(100, 100, 1100, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        layout = QVBoxLayout()
        
        self.btn_sec = QPushButton('🖼️ Resim Seç ve Derinlik Tahmin Et')
        self.btn_sec.setFixedHeight(50)
        self.btn_sec.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; border-radius: 10px;")
        self.btn_sec.clicked.connect(self.analiz_et)
        
        display_layout = QHBoxLayout()
        self.lbl_sol = self.create_display("Orijinal Resim")
        self.lbl_sag = self.create_display("Derinlik Haritası (Depth Map)")
        
        display_layout.addWidget(self.lbl_sol)
        display_layout.addWidget(self.lbl_sag)
        
        layout.addWidget(self.btn_sec)
        layout.addLayout(display_layout)
        self.setLayout(layout)

    def create_display(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("border: 2px solid #34495e; background-color: #2c3e50;")
        lbl.setMinimumSize(500, 400)
        return lbl

    def analiz_et(self):
        yol, _ = QFileDialog.getOpenFileName(self, "Resim Seç")
        if yol:
            img = cv2.imread(yol)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            pix = QPixmap(yol)
            self.lbl_sol.setPixmap(pix.scaled(500, 400, Qt.KeepAspectRatio))
            
            input_batch = self.transform(img_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            
            output_norm = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)
            
            h, w, ch = output_color.shape
            q_img = QImage(output_color.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped().copy()
            self.lbl_sag.setPixmap(QPixmap.fromImage(q_img).scaled(500, 400, Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DepthEstimationApp()
    ex.show()
    sys.exit(app.exec_())