import sys 
import os 
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
    print(f"MiDaS model: {midas_model_type} loaded.")
except Exception as e:
    print(f"Error: Could not load MiDaS model. Estimation disabled. {e}")
    midas = None 

class DepthEstimationApp(QWidget):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Derinlik Tahmini Arayüzü")
        self.setGeometry(100, 100, 1000, 600)


        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        image_layout = QHBoxLayout()


        self.btn_load_image = QPushButton("1. Resmi Seç")
        self.btn_run = QPushButton("2. Derinlik Tahminini Çalıştır")
        self.btn_export_3d = QPushButton("3. 3D Nokta Bulutu Olarak Kaydet (PLY)") 
        
        button_layout.addWidget(self.btn_load_image)
        button_layout.addWidget(self.btn_run)
        button_layout.addWidget(self.btn_export_3d)

        self.label_original = QLabel("Orijinal Resim (Input)")
        self.label_depth = QLabel("Derinlik Haritası (Output)")
        
        self.label_original.setFixedSize(480, 480)
        self.label_depth.setFixedSize(480, 480)
        
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_depth)

        main_layout.addLayout(button_layout)
        main_layout.addLayout(image_layout)
        self.setLayout(main_layout)

        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_run.clicked.connect(self.run_depth_estimation) 
        self.btn_export_3d.clicked.connect(self.export_3d_model)


        self.image_path = None
        self.model = midas
        self.raw_depth_map = None      
        self.original_image_np = None  
        
        if self.model:
            self.model.eval() 
            self.device = torch.device("cpu") 
            print("Model is ready.")
        else:
             self.device = None
             self.btn_run.setEnabled(False) 
             print("Model unavailable, run button disabled.")

        self.btn_export_3d.setEnabled(False)


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg)")
        
        if file_path:
            self.image_path = file_path
            

            pixmap = QPixmap(file_path).scaled(self.label_original.width(), self.label_original.height())
            self.label_original.setPixmap(pixmap)
            print(f"Image loaded: {file_path}")
            self.btn_export_3d.setEnabled(False) 

    def run_depth_estimation(self):
        if self.image_path is None or self.model is None:
            print("Error: Select an image first.")
            return

        image = Image.open(self.image_path).convert("RGB")
        
        self.original_image_np = np.array(image)

        transform = T.Compose([
            T.Resize(384, interpolation=InterpolationMode.BICUBIC), 
            T.CenterCrop(384),
            T.ToTensor(),  
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        input_tensor = transform(image).unsqueeze(0).to(self.device)


        with torch.no_grad(): 
            prediction = self.model(input_tensor)
            
    
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1], 
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        self.raw_depth_map = prediction.cpu().numpy()


        depth_map_normalized = (self.raw_depth_map - self.raw_depth_map.min()) / (self.raw_depth_map.max() - self.raw_depth_map.min())
        depth_map_display = (depth_map_normalized * 255).astype(np.uint8)
        

        depth_image = Image.fromarray(depth_map_display, mode='L') 
        qimage = QImage(depth_image.tobytes(), depth_image.width, depth_image.height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage).scaled(self.label_depth.width(), self.label_depth.height())
        
        self.label_depth.setPixmap(pixmap)
        print("Depth estimation completed.")

        self.btn_export_3d.setEnabled(True)
        

    def export_3d_model(self):
        if self.raw_depth_map is None or self.original_image_np is None:
            print("Error: Run depth estimation before exporting 3D model.")
            return
        
        H, W = self.raw_depth_map.shape
        
      
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        

        Z = self.raw_depth_map
        
        N = W * H
        

        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        colors = self.original_image_np.reshape(N, 3)
        

        point_cloud = np.hstack([points, colors]).astype(np.float32)
        
  
        output_file, _ = QFileDialog.getSaveFileName(self, "3D Nokta Bulutunu Kaydet", "output.ply", "PLY Files (*.ply)")
        
        if not output_file:
            print("3D export cancelled.")
            return


        header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        

        data_lines = [f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(p[3])} {int(p[4])} {int(p[5])}" 
                      for p in point_cloud]
        
    
        try:
            with open(output_file, 'w') as f:
                f.write(header)
                f.write('\n'.join(data_lines))
            print(f"3D Point Cloud successfully saved to: {output_file}")
        except Exception as e:
            print(f"Error: Could not save PLY file: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DepthEstimationApp()
    window.show()
    sys.exit(app.exec_())
