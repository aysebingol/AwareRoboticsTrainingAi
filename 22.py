import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ResimArayuz(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('YOLOv5 Model Sonuç Ekranı')
        self.setGeometry(100, 100, 600, 500)

        layout = QVBoxLayout()

        self.label_yazi = QLabel('Eğitim Sonucu: results.png', self)
        self.label_yazi.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_yazi)

        self.label_resim = QLabel(self)
        
        pixmap = QPixmap('/home/aysebingol/Desktop/ROS/AwareRoboticsTrainingAi/results.png') 
        
        self.label_resim.setPixmap(pixmap.scaled(550, 400, Qt.KeepAspectRatio))
        self.label_resim.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_resim)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pencere = ResimArayuz()
    pencere.show()
    sys.exit(app.exec_())