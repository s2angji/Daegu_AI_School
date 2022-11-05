import sys
import os
import glob
from natsort import natsorted

import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from math import ceil
from PIL import Image
import threading


class Communicate(QObject):
    stop_flag = True
    pause_flag = True

    start_signal = pyqtSignal()
    draw_signal = pyqtSignal(dict)
    stop_signal = pyqtSignal()


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        # 스레드
        self.t = None

        # 탭페이지에서 캡쳐 버튼을 누르면 GRAY 및 BLUR 프로세싱을 거친 이미지를 저장하는 변수
        self.captured_imgs = {'rgb': None, 'gray': None, 'blur': None}

        # 사용자 정의 시그널
        def start():
            self.signal.stop_flag = False
            self.signal.pause_flag = False

            for tab in [self.tab_single, self.tab_triple]:
                tab.start_btn.setEnabled(False)
                tab.stop_btn.setEnabled(True)
                tab.cap_btn.setEnabled(True)
                tab.cap_cancel_btn.setEnabled(False)
                tab.save_btn.setEnabled(False)

                for label in tab.labels:
                    label[0].setEnabled(True)
        # 활성화된 탭의(active_tab) 모든 레이블에 웹캠의 한 프레임을 그림
        def draw(pixs):
            for i, label in enumerate(self.active_tab.labels):
                label[0].setPixmap(pixs[label[1]].scaled(label[0].width(), label[0].height(), Qt.KeepAspectRatio))

        def stop():
            self.signal.stop_flag = True
            self.signal.pause_flag = True

            for tab in [self.tab_single, self.tab_triple]:
                tab.start_btn.setEnabled(True)
                tab.stop_btn.setEnabled(False)
                tab.cap_btn.setEnabled(False)
                tab.cap_cancel_btn.setEnabled(False)
                tab.save_btn.setEnabled(False)

                for label in tab.labels:
                    label[0].setEnabled(False)

        self.signal = Communicate()
        self.signal.start_signal.connect(start)
        self.signal.draw_signal.connect(draw)
        self.signal.stop_signal.connect(stop)

        # 현재 활성화된 탭
        self.active_tab = None
        # 탭 변경시 활성화된 탭을 변경
        def on_current_changed(idx):
            self.active_tab = [self.tab_single, self.tab_triple][idx]
        # 탭 위젯에 탭 페이지를 등록
        # 탭 페이지의 생성자 매개변수의 labels는
        # 웹캠 영상을 뿌려줄 QLabel 인스턴스와
        # 어떤 이미지 프로세싱을(RGB, GRAY 또는 BLUR) 원하는지 쌍으로 결정한다
        # 웹캠 영상을 뿌려줄 QLabel 인스턴스는 자동으로 GridLayout에 배치 되도록 만들었으므로, 원하는 쌍을 더 많이 추가해도 괜찮다
        tabs = QTabWidget()
        tabs.currentChanged.connect(on_current_changed)
        self.tab_single = TabPage(self, labels=((QLabel(), 'rgb'),))
        self.tab_triple = TabPage(self, labels=((QLabel(), 'rgb'), (QLabel(), 'gray'), (QLabel(), 'blur'),))
        tabs.addTab(self.tab_single, 'Single')
        tabs.addTab(self.tab_triple, 'Triple')

        vbox = QVBoxLayout()
        vbox.addWidget(tabs)
        self.setLayout(vbox)

        # MainWindow의 크기 조정 및 화면 중앙에 띄워지도록 만들기
        self.setWindowTitle('Camera Capture')
        rect = QDesktopWidget().availableGeometry()
        self.resize(rect.width() * 0.55, rect.height() * 0.75)
        rect = self.frameGeometry()
        rect.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(rect.topLeft())

        # 처음 시작시 버튼들의 활성화 유무를 결정하기 위해서 호출
        self.signal.stop_signal.emit()


class TabPage(QWidget):

    def __init__(self, main, labels):
        super().__init__()
        self.main = main
        self.labels = labels

        def on_clicked():
            # 웹캠 정지 버튼
            if self.sender() is self.stop_btn:
                self.main.signal.stop_signal.emit()
                # 스레드의 종료를 기다려 줌
                self.main.t.join()
            # 캡처 버튼 & 캡처 취소 버튼
            elif self.sender() in [self.cap_btn, self.cap_cancel_btn]:
                self.main.signal.pause_flag = self.sender() is self.cap_btn

                self.cap_btn.setEnabled(self.sender() is self.cap_cancel_btn)
                self.cap_cancel_btn.setEnabled(self.sender() is self.cap_btn)
                self.save_btn.setEnabled(self.sender() is self.cap_btn)
            # 저장 버튼
            elif self.sender() is self.save_btn:
                if self.edit.text().strip() == '':
                    print('라벨 이름이 없습니다.')
                    return

                for label in labels:
                    if self.cbox.currentIndex() == 0:
                        self.cbox.addItem(self.edit.text())
                        self.cbox.setCurrentText(self.edit.text())

                    # natsort 활용하여 정렬 후, 마지막 경로명의 숫자를 파악
                    paths = natsorted(glob.glob(os.path.join(os.getcwd(), f'image_{label[1]}_{self.cbox.currentText()}_*.png')))
                    if len(paths) == 0:
                        save_idx = 0
                    else:
                        save_idx = int(paths[-1].split('_')[-1][:-4]) + 1

                    Image.fromarray(cv2.cvtColor(self.main.captured_imgs[label[1]], cv2.COLOR_BGR2RGB)) \
                        .save(f'image_{label[1]}_{self.cbox.currentText()}_{save_idx}.png')

                    self.main.signal.pause_flag = False

                    self.cap_btn.setEnabled(True)
                    self.cap_cancel_btn.setEnabled(False)
                    self.save_btn.setEnabled(False)
            # 웹캠 켜기 버튼
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

                        # 이미지 프로세싱 RGB
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.main.captured_imgs['rgb'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        q_img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                        rgb_pix = QPixmap.fromImage(q_img)
                        # 이미지 프로세싱 GRAY
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self.main.captured_imgs['gray'] = img
                        q_img = QImage(img, img.shape[1], img.shape[0], QImage.Format_Grayscale8)
                        gray_pix = QPixmap.fromImage(q_img)
                        # 이미지 프로세싱 BLUR
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = cv2.blur(img, (20, 20))
                        self.main.captured_imgs['blur'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        q_img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                        blur_pix = QPixmap.fromImage(q_img)

                        # MainWindow(UI 스레드)에게 준비된 이미지를 Draw하라고 신호를 보냄
                        self.main.signal.draw_signal.emit({'rgb': rgb_pix, 'gray': gray_pix, 'blur': blur_pix})

                    cap.release()

                self.main.t = threading.Thread(target=run)
                self.main.t.daemon = True
                self.main.t.start()

        # 탭페이지 인스턴스 생성시 받은 labels 정보로 GridLayout에 label들을 자동으로 배치
        self.hbox = QHBoxLayout()
        grid = QGridLayout()
        s = ceil(len(labels) ** 0.5)    # 사각형 한 변의 길이
        pos = [(x, y) for x in range(s) for y in range(s) if x * s + (y + 1) <= len(labels)]    # 배치될 위치 정보 (x, y)
        for label, p in zip(labels, pos):   # 레이블을 GridLayout에 배치
            label[0].setAlignment(Qt.AlignCenter)
            grid.addWidget(label[0], *p)
        self.hbox.addLayout(grid, 1)

        # 각종 버튼들
        vbox = QVBoxLayout()
        self.start_btn = QPushButton('Camera Open')
        self.stop_btn = QPushButton('Camera Close')
        self.cap_btn = QPushButton('Capture')
        self.cap_cancel_btn = QPushButton('Capture Cancel')
        self.save_btn = QPushButton('Save')
        self.start_btn.clicked.connect(on_clicked)
        self.stop_btn.clicked.connect(on_clicked)
        self.cap_btn.clicked.connect(on_clicked)
        self.cap_cancel_btn.clicked.connect(on_clicked)
        self.save_btn.clicked.connect(on_clicked)
        vbox.addWidget(self.start_btn)
        vbox.addWidget(self.stop_btn)
        vbox.addWidget(self.cap_btn)
        vbox.addWidget(self.cap_cancel_btn)
        vbox.addWidget(self.save_btn)
        vbox.addStretch(1)

        # 이미지의 라벨 정보를 지정하기 위한 LineEdit과 ComboBox
        # LineEdit은 텍스트 변경시 마다 기존에 사용한 라벨이 ComboBox 아이템에 저장되어 있는지 찾는다
        def on_text_changed():
            if self.cbox.findText(self.edit.text()) == -1 and self.cbox.currentIndex() != 0:
                self.cbox.setCurrentIndex(0)
                return

            if self.cbox.findText(self.edit.text()) != self.cbox.currentIndex():
                self.cbox.setCurrentText(self.edit.text())

        vbox.addWidget(QLabel('Label :'))
        self.edit = QLineEdit()
        self.edit.textChanged.connect(on_text_changed)
        vbox.addWidget(self.edit)
        # 사용자가 ComboBox에 있는 아이템으로 라벨을 지정하면 LineEdit도 동일하게 변경되도록 한다
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
        self.hbox.addLayout(vbox)
        self.setLayout(self.hbox)

        self.cbox.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
