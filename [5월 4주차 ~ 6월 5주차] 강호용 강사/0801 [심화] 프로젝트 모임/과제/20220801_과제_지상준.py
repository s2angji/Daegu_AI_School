import sys
import os
import glob
from natsort import natsorted

import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from numpy import ndarray
from PIL import Image
import threading


class Communicate(QObject):
    stop_flag = True
    pause_flag = True

    start_signal = pyqtSignal()
    draw_signal = pyqtSignal(ndarray)
    stop_signal = pyqtSignal()


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        def start():
            self.signal.stop_flag = False
            self.signal.pause_flag = False

            for tab in [self.single, self.triple]:
                tab.start_btn.setEnabled(False)
                tab.stop_btn.setEnabled(True)
                tab.cap_btn.setEnabled(True)
                tab.cap_cancel_btn.setEnabled(False)
                tab.save_btn.setEnabled(False)

            self.single.label1.setEnabled(True)
            self.triple.label1.setEnabled(True)
            self.triple.label2.setEnabled(True)
            self.triple.label3.setEnabled(True)

        def draw(img):
            self.captured_img = img.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            self.active_tab.label1.setPixmap(pix.scaled(self.active_tab.label1.width(), self.active_tab.label1.height(), Qt.KeepAspectRatio))

            if self.active_tab is self.triple:
                img = cv2.cvtColor(self.captured_img.copy(), cv2.COLOR_BGR2GRAY)
                q_img = QImage(img, img.shape[1], img.shape[0], QImage.Format_Grayscale8)
                pix = QPixmap.fromImage(q_img)
                self.active_tab.label2.setPixmap(pix.scaled(self.active_tab.label2.width(), self.active_tab.label2.height(), Qt.KeepAspectRatio))
                img = cv2.cvtColor(self.captured_img.copy(), cv2.COLOR_BGR2RGB)
                img = cv2.blur(img, (20, 20))
                q_img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(q_img)
                self.active_tab.label3.setPixmap(pix.scaled(self.active_tab.label3.width(), self.active_tab.label3.height(), Qt.KeepAspectRatio))

        def stop():
            self.signal.stop_flag = True
            self.signal.pause_flag = True

            for tab in [self.single, self.triple]:
                tab.start_btn.setEnabled(True)
                tab.stop_btn.setEnabled(False)
                tab.cap_btn.setEnabled(False)
                tab.cap_cancel_btn.setEnabled(False)
                tab.save_btn.setEnabled(False)

            self.single.label1.setEnabled(False)
            self.triple.label1.setEnabled(False)
            self.triple.label2.setEnabled(False)
            self.triple.label3.setEnabled(False)

        self.signal = Communicate()
        self.signal.start_signal.connect(start)
        self.signal.draw_signal.connect(draw)
        self.signal.stop_signal.connect(stop)

        self.active_tab = None

        def on_current_changed(idx):
            self.active_tab = [self.single, self.triple][idx]
            self.active_tab.stop_btn.click()

        tabs = QTabWidget()
        tabs.currentChanged.connect(on_current_changed)
        self.single = SingleTab(self)
        self.triple = TripleTab(self)
        tabs.addTab(self.single, 'Single')
        tabs.addTab(self.triple, 'Triple')

        vbox = QVBoxLayout()
        vbox.addWidget(tabs)
        self.setLayout(vbox)

        self.setWindowTitle('Camera Capture')
        rect = QDesktopWidget().availableGeometry()
        self.resize(rect.width() * 0.8, rect.height() * 0.7)
        rect = self.frameGeometry()
        rect.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(rect.topLeft())


class SingleTab(QWidget):

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.t = None

        def on_clicked():

            if self.sender() is self.stop_btn:
                self.main.signal.stop_signal.emit()
                if self.t is not None:
                    self.t.join()

            elif self.sender() is self.cap_btn:
                self.main.signal.pause_flag = True

                self.cap_btn.setEnabled(False)
                self.cap_cancel_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

            elif self.sender() is self.cap_cancel_btn:
                self.main.signal.pause_flag = False

                self.cap_btn.setEnabled(True)
                self.cap_cancel_btn.setEnabled(False)
                self.save_btn.setEnabled(False)

            elif self.sender() is self.save_btn:
                if self.edit.text().strip() == '':
                    return

                if self.cbox.currentIndex() == 0:
                    self.cbox.addItem(self.edit.text())
                    self.cbox.setCurrentText(self.edit.text())

                paths = natsorted(glob.glob(os.path.join(os.getcwd(), f'image_{self.cbox.currentText()}_*.png')))
                if len(paths) == 0:
                    save_idx = 0
                else:
                    save_idx = int(paths[-1].split('_')[-1][:-4]) + 1

                Image.fromarray(cv2.cvtColor(self.main.captured_img, cv2.COLOR_BGR2RGB))\
                    .save(f'image_{self.cbox.currentText()}_{save_idx}.png')

                self.main.signal.pause_flag = False

                self.cap_btn.setEnabled(True)
                self.cap_cancel_btn.setEnabled(False)
                self.save_btn.setEnabled(False)

            elif self.sender() is self.start_btn:
                def run():
                    self.main.signal.start_signal.emit()

                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        print('카메라 열기 실패')
                        exit()

                    while not self.main.signal.stop_flag:
                        if self.main.signal.pause_flag:
                            continue

                        ret, frame = cap.read()
                        if not ret:
                            print('read() 실패')
                            break
                        self.main.signal.draw_signal.emit(frame)

                    cap.release()

                self.t = threading.Thread(target=run)
                self.t.daemon = True
                self.t.start()

        self.hbox = QHBoxLayout()
        self.label1 = QLabel()
        self.hbox.addWidget(self.label1, 2)

        vbox = QVBoxLayout()
        self.start_btn = QPushButton('Camera Open')
        self.start_btn.clicked.connect(on_clicked)
        self.stop_btn = QPushButton('Camera Close')
        self.stop_btn.clicked.connect(on_clicked)
        self.cap_btn = QPushButton('Capture')
        self.cap_btn.clicked.connect(on_clicked)
        self.cap_cancel_btn = QPushButton('Capture Cancel')
        self.cap_cancel_btn.clicked.connect(on_clicked)
        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(on_clicked)
        vbox.addWidget(self.start_btn)
        vbox.addWidget(self.stop_btn)
        vbox.addWidget(self.cap_btn)
        vbox.addWidget(self.cap_cancel_btn)
        vbox.addWidget(self.save_btn)
        vbox.addStretch(1)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cap_btn.setEnabled(False)
        self.cap_cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        def on_text_changed():
            if self.cbox.findText(self.edit.text()) == -1 and self.cbox.currentIndex() != 0:
                self.cbox.setCurrentIndex(0)
                return

            if self.cbox.findText(self.edit.text()) != self.cbox.currentIndex():
                self.cbox.setCurrentText(self.edit.text())
                return

        vbox.addWidget(QLabel('Label :'))
        self.edit = QLineEdit()
        self.edit.textChanged.connect(on_text_changed)
        vbox.addWidget(self.edit)

        def on_current_text_changed():
            if self.cbox.currentIndex() == 0:
                return

            if self.cbox.currentText() != self.edit.text():
                self.edit.setText(self.cbox.currentText())

        self.cbox = QComboBox()
        self.cbox.currentTextChanged.connect(on_current_text_changed)
        self.cbox.addItems(['', 'happy', 'anger', 'normal'])
        vbox.addWidget(self.cbox)
        vbox.addStretch(3)
        self.hbox.addLayout(vbox, 1)
        self.setLayout(self.hbox)

        self.cbox.setCurrentIndex(1)


class TripleTab(QWidget):

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.t = None

        def on_clicked():

            if self.sender() is self.stop_btn:
                self.main.signal.stop_signal.emit()
                if self.t is not None:
                    self.t.join()

            elif self.sender() is self.cap_btn:
                self.main.signal.pause_flag = True

                self.cap_btn.setEnabled(False)
                self.cap_cancel_btn.setEnabled(True)
                self.save_btn.setEnabled(True)

            elif self.sender() is self.cap_cancel_btn:
                self.main.signal.pause_flag = False

                self.cap_btn.setEnabled(True)
                self.cap_cancel_btn.setEnabled(False)
                self.save_btn.setEnabled(False)

            elif self.sender() is self.save_btn:
                if self.edit.text().strip() == '':
                    return

                if self.cbox.currentIndex() == 0:
                    self.cbox.addItem(self.edit.text())
                    self.cbox.setCurrentText(self.edit.text())

                paths = natsorted(glob.glob(os.path.join(os.getcwd(), f'image_{self.cbox.currentText()}_*.png')))
                if len(paths) == 0:
                    save_idx = 0
                else:
                    save_idx = int(paths[-1].split('_')[-1][:-4]) + 1

                Image.fromarray(cv2.cvtColor(self.main.captured_img, cv2.COLOR_BGR2RGB))\
                    .save(f'image_{self.cbox.currentText()}_{save_idx}.png')

                self.main.signal.pause_flag = False

                self.cap_btn.setEnabled(True)
                self.cap_cancel_btn.setEnabled(False)
                self.save_btn.setEnabled(False)

            elif self.sender() is self.start_btn:
                def run():
                    self.main.signal.start_signal.emit()

                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        print('카메라 열기 실패')
                        exit()

                    while not self.main.signal.stop_flag:
                        if self.main.signal.pause_flag:
                            continue

                        ret, frame = cap.read()
                        if not ret:
                            print('read() 실패')
                            break
                        self.main.signal.draw_signal.emit(frame)

                    cap.release()

                self.t = threading.Thread(target=run)
                self.t.daemon = True
                self.t.start()

        self.hbox = QHBoxLayout()
        grid = QGridLayout()
        self.label1 = QLabel()
        self.label2 = QLabel()
        self.label3 = QLabel()
        grid.addWidget(self.label1, 0, 0)
        grid.addWidget(self.label2, 0, 1)
        grid.addWidget(self.label3, 1, 0)
        self.hbox.addLayout(grid, 2)

        vbox = QVBoxLayout()
        self.start_btn = QPushButton('Camera Open')
        self.start_btn.clicked.connect(on_clicked)
        self.stop_btn = QPushButton('Camera Close')
        self.stop_btn.clicked.connect(on_clicked)
        self.cap_btn = QPushButton('Capture')
        self.cap_btn.clicked.connect(on_clicked)
        self.cap_cancel_btn = QPushButton('Capture Cancel')
        self.cap_cancel_btn.clicked.connect(on_clicked)
        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(on_clicked)
        vbox.addWidget(self.start_btn)
        vbox.addWidget(self.stop_btn)
        vbox.addWidget(self.cap_btn)
        vbox.addWidget(self.cap_cancel_btn)
        vbox.addWidget(self.save_btn)
        vbox.addStretch(1)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cap_btn.setEnabled(False)
        self.cap_cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        def on_text_changed():
            if self.cbox.findText(self.edit.text()) == -1 and self.cbox.currentIndex() != 0:
                self.cbox.setCurrentIndex(0)
                return

            if self.cbox.findText(self.edit.text()) != self.cbox.currentIndex():
                self.cbox.setCurrentText(self.edit.text())
                return

        vbox.addWidget(QLabel('Label :'))
        self.edit = QLineEdit()
        self.edit.textChanged.connect(on_text_changed)
        vbox.addWidget(self.edit)

        def on_current_text_changed():
            if self.cbox.currentIndex() == 0:
                return

            if self.cbox.currentText() != self.edit.text():
                self.edit.setText(self.cbox.currentText())

        self.cbox = QComboBox()
        self.cbox.currentTextChanged.connect(on_current_text_changed)
        self.cbox.addItems(['', 'happy', 'anger', 'normal'])
        vbox.addWidget(self.cbox)
        vbox.addStretch(3)
        self.hbox.addLayout(vbox, 1)
        self.setLayout(self.hbox)

        self.cbox.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
