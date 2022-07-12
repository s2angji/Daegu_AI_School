import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from main_window import MainWindow


class ClassificationWidget(QWidget):

    def __init__(self, main):
        super().__init__()

        edit1 = QTextEdit('분류 문제를 !! 여기서 !!')
        edit2 = QTextEdit('내용을 입력 !!!')
        edit3 = QTextEdit('디버깅 !!!')
        vbox = QVBoxLayout()
        vbox.addWidget(edit1)
        vbox.addWidget(edit2)
        vbox.addWidget(edit3)
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.set_func_widget(1, ClassificationWidget)
    mainWindow.show()
    sys.exit(app.exec_())
