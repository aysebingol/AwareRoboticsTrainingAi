import os
import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import open3d as o3d 

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

        self.last_depth_map = None 
        self.original_image_rgb = None 

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Aware Robotics - Derinlik Tahmini ve 3D Modelleme')
        self.setGeometry(100, 100, 1200, 650)
        self.setStyleSheet("background-color: #1a1a2e; color: #e0e0e0;")

        main_layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        self.btn_sec = QPushButton('🖼️ Resim Seç ve Derinlik Tahmin Et')
        self.btn_sec.setFixedHeight(50)
        self.btn_sec.setStyleSheet("""
            QPushButton { background-color: #6a1b9a; color: white; font-weight: bold; border-radius: 10px; font-size: 16px; }
            QPushButton:hover { background-color: #8e24aa; }
        """)
        self.btn_sec.clicked.connect(self.analiz_et)
        
        self.btn_3d_model = QPushButton('✨ 3D Model Oluştur (.ply)')
        self.btn_3d_model.setFixedHeight(50)
        self.btn_3d_model.setStyleSheet("""
            QPushButton { background-color: #388e3c; color: white; font-weight: bold; border-radius: 10px; font-size: 16px; }
            QPushButton:hover { background-color: #43a047; }
        """)
        self.btn_3d_model.clicked.connect(self.create_3d_model)
        self.btn_3d_model.setEnabled(False) 
        
        button_layout.addWidget(self.btn_sec)
        button_layout.addWidget(self.btn_3d_model)
        main_layout.addLayout(button_layout)
        
        display_layout = QHBoxLayout()
        self.lbl_sol = self.create_display("Orijinal Resim")
        self.lbl_sag = self.create_display("Derinlik Haritası (Heatmap)")
        
        display_layout.addWidget(self.lbl_sol)
        display_layout.addWidget(self.lbl_sag)
        
        main_layout.addLayout(display_layout)

        self.lbl_durum = QLabel("Hazır. Bir resim seçerek başlayın.")
        self.lbl_durum.setAlignment(Qt.AlignCenter)
        self.lbl_durum.setStyleSheet("color: #bbdefb; font-size: 14px; margin-top: 10px;")
        main_layout.addWidget(self.lbl_durum)

        self.setLayout(main_layout)

    def create_display(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("border: 2px solid #4a148c; background-color: #2e104a; border-radius: 5px;")
        lbl.setMinimumSize(550, 450)
        return lbl

    def analiz_et(self):
        yol, _ = QFileDialog.getOpenFileName(self, "Resim Seç")
        if yol:
            self.lbl_durum.setText("Resim yükleniyor ve derinlik tahmin ediliyor...")
            QApplication.processEvents()

            img = cv2.imread(yol)
            if img is None:
                QMessageBox.warning(self, "Hata", "Geçersiz resim dosyası!")
                self.lbl_durum.setText("Hata: Resim yüklenemedi.")
                return

            self.original_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            
            pix = QPixmap(yol)
            self.lbl_sol.setPixmap(pix.scaled(550, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            try:
                input_batch = self.transform(self.original_image_rgb).to(self.device)

                with torch.no_grad():
                    prediction = self.midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=self.original_image_rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                output = prediction.cpu().numpy()
                self.last_depth_map = output 

                output_norm = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)
                
                h, w, ch = output_color.shape
                q_img = QImage(output_color.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped().copy()
                self.lbl_sag.setPixmap(QPixmap.fromImage(q_img).scaled(550, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                self.lbl_durum.setText("Derinlik tahmini tamamlandı. Şimdi 3D model oluşturabilirsiniz.")
                self.btn_3d_model.setEnabled(True) 
                
            except Exception as e:
                self.lbl_durum.setText(f"Derinlik Tahmin Hatası: {e}")
                self.btn_3d_model.setEnabled(False)

    def create_3d_model(self):
        if self.last_depth_map is None or self.original_image_rgb is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir resim analiz etmelisiniz!")
            return

        self.lbl_durum.setText("3D model oluşturuluyor...")
        QApplication.processEvents()

        try:
            h, w = self.last_depth_map.shape
            
            f = 500
            cx, cy = w / 2, h / 2

            points = []
            colors = []

            for v in range(h):
                for u in range(w):
                    depth = self.last_depth_map[v, u]
                    
                    z = 1.0 / (depth + 1e-6)
                    
                    x = (u - cx) * z / f
                    y = (v - cy) * z / f
                    
                    points.append([x, y, z])
                    colors.append(self.original_image_rgb[v, u] / 255.0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

            save_path, _ = QFileDialog.getSaveFileName(self, "3D Model Kaydet", "", "PLY Dosyaları (*.ply)")
            if save_path:
                o3d.io.write_point_cloud(save_path, pcd)
                self.lbl_durum.setText(f"3D model '{os.path.basename(save_path)}' başarıyla kaydedildi!")
                QMessageBox.information(self, "Başarılı", f"3D model '{os.path.basename(save_path)}' kaydedildi!\nBir 3D görüntüleyici ile açabilirsiniz.")
            else:
                self.lbl_durum.setText("3D model kaydetme iptal edildi.")

        except Exception as e:
            self.lbl_durum.setText(f"3D Model Oluşturma Hatası: {e}")
            QMessageBox.critical(self, "Hata", f"3D model oluşturulamadı: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DepthEstimationApp()
    ex.show()
    sys.exit(app.exec_())