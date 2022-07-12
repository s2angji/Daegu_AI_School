import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from main_window import MainWindow
from classification_window import ClassificationWidget
from object_detection_window import ObjectDetectionWidget


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.set_func_widget(1, ClassificationWidget)
    mainWindow.set_func_widget(2, ObjectDetectionWidget)
    mainWindow.show()
    sys.exit(app.exec_())
