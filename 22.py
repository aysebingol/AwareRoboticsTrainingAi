import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("PyQt5 resim gösterme")
window.setGeometry(100, 100, 400, 300)

label = QLabel(window)
pixmap = QPixmap("lacoste.png")
label.setPixmap(pixmap)

label.resize(pixmap.width(), pixmap.height())

window.show()

sys.exit(app.exec_())