import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class WelcomeHomeWidget(QWidget):

    def __init__(self, main):
        super().__init__()

        btn = QPushButton()
        btn.setFixedHeight(QDesktopWidget().availableGeometry().height() * 0.75)
        btn.setStyleSheet('border-image: url(./img/home.png); border-radius: 15px;')
        btn.clicked.connect(lambda: main.debug('환영합니다.'))
        vbox = QVBoxLayout()
        vbox.addWidget(btn)
        self.setLayout(vbox)


class WelcomeClassificationWidget(QWidget):

    def __init__(self, main):
        super().__init__()

        btn = QPushButton()
        btn.setFixedHeight(QDesktopWidget().availableGeometry().height() * 0.75)
        btn.setStyleSheet('border-image: url(./img/classification.png); border-radius: 15px;')
        btn.clicked.connect(lambda: main.insert_func_widget(main.widget_info[1][1]))
        vbox = QVBoxLayout()
        vbox.addWidget(btn)
        self.setLayout(vbox)


class WelcomeObjectDetectionWidget(QWidget):

    def __init__(self, main):
        super().__init__()

        btn = QPushButton()
        btn.setFixedHeight(QDesktopWidget().availableGeometry().height() * 0.75)
        btn.setStyleSheet('border-image: url(./img/object_detection.jpg); border-radius: 15px;')
        btn.clicked.connect(lambda: main.insert_func_widget(main.widget_info[2][1]))
        vbox = QVBoxLayout()
        vbox.addWidget(btn)
        self.setLayout(vbox)


class MainWindow(QMainWindow):

    def debug(self, message: str):
        self.status.setText('    ' + message)

    def set_start_widget(self, idx: int, widget):
        self.widget_info[idx][0] = widget

    def set_func_widget(self, idx: int, widget):
        self.widget_info[idx][1] = widget

    def remove_widget(self, widget: QWidget):
        widget.deleteLater()

    def insert_start_widget(self, widget):
        if widget is None:
            return

        if self.active_widget is not None:
            self.remove_widget(self.active_widget)
        self.active_widget = widget(self)
        self.contentVBox.insertWidget(2, self.active_widget)

    def insert_func_widget(self, widget):
        if widget is None:
            return

        if self.active_widget is not None:
            self.remove_widget(self.active_widget)
        self.active_widget = widget(self)
        self.contentVBox.insertWidget(2, self.active_widget)

    def __init__(self):
        super().__init__()

        # 메뉴 선택시 처음 진입점이 되는 위젯, 주요 기능을 맡은 위젯 및 처음 소개말
        self.widget_info = [[None, None, '안녕하세요 :)'],            # Home
                            [None, None, '분류를 시작합니다 :)'],      # Classification
                            [None, None, '객체 감지를 시작합니다 :)']]  # Object Detection
        self.set_start_widget(0, WelcomeHomeWidget)
        self.set_start_widget(1, WelcomeClassificationWidget)
        self.set_start_widget(2, WelcomeObjectDetectionWidget)
        # 현재 contentVBox의 2번째 위젯 인스턴스
        self.active_widget = None

        # 타이틀바 없애기 및 드래그로 창 이동 구현을 위한 인스턴스 변수
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.drag_flag, self.drag_pos = False, None

        # 창 사이즈 조절 및 화면 가운데 위치
        rect = QDesktopWidget().availableGeometry()
        self.resize(rect.width() * 0.8, rect.height() * 0.8)
        rect = self.frameGeometry()
        rect.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(rect.topLeft())

        # 메인 위젯
        self.mainWidget = QWidget(objectName='mainWidget')
        self.mainWidget.setStyleSheet('''
        #mainWidget {
            border-radius: 15px;
            background-color: #DDDDDD;
        }''')
        self.setCentralWidget(self.mainWidget)
        layout = QHBoxLayout()

        # 메뉴 버튼
        menuVBox = QVBoxLayout()
        self.menuBtn = []
        self.menuLbl = []
        self.menuBox = []
        self.menuBtn.append(QPushButton(' Home', objectName='BtnHome'))            # Home 버튼
        # icon = QIcon()
        # icon.addPixmap(QPixmap('./img/1.png'))
        # self.menuBtn[0].setIcon(icon)
        self.menuBtn[0].setFixedHeight(70)
        self.menuBtn[0].clicked.connect(self.menuClick)
        self.menuLbl.append(QLabel())
        self.menuLbl[0].setFixedWidth(15)
        self.menuBox.append(QHBoxLayout())
        self.menuBox[0].addWidget(self.menuLbl[0])
        self.menuBox[0].addWidget(self.menuBtn[0])
        self.menuBtn.append(QPushButton(' Classification', objectName='BtnClassification'))  # Classification 버튼
        # icon = QIcon()
        # icon.addPixmap(QPixmap('./img/2.png'))
        # self.menuBtn[1].setIcon(icon)
        self.menuBtn[1].setFixedHeight(70)
        self.menuBtn[1].clicked.connect(self.menuClick)
        self.menuLbl.append(QLabel())
        self.menuLbl[1].setFixedWidth(15)
        self.menuBox.append(QHBoxLayout())
        self.menuBox[1].addWidget(self.menuLbl[1])
        self.menuBox[1].addWidget(self.menuBtn[1])
        self.menuBtn.append(QPushButton(' Object Detection   ', objectName='BtnObjectDetection'))  # Object Detection 버튼
        # icon = QIcon()
        # icon.addPixmap(QPixmap('./img/1.png'))
        # self.menuBtn[0].setIcon(icon)
        self.menuBtn[2].setFixedHeight(70)
        self.menuBtn[2].clicked.connect(self.menuClick)
        self.menuLbl.append(QLabel())
        self.menuLbl[2].setFixedWidth(15)
        self.menuBox.append(QHBoxLayout())
        self.menuBox[2].addWidget(self.menuLbl[2])
        self.menuBox[2].addWidget(self.menuBtn[2])
        btnQuestion = QPushButton('  ?', objectName='BtnQuestion')
        btnQuestion.setFixedWidth(90)
        btnQuestion.setFixedHeight(60)
        btnQuestion.clicked.connect(lambda: QMessageBox.about(self, 'About', '대구 AI School 1팀<br><br>'
                                                                             '팀장: 지상준<br>'
                                                                             '팀원: 김혜린, 시민주'))
        menuVBox.addStretch(2)
        menuVBox.addLayout(self.menuBox[0])
        menuVBox.addLayout(self.menuBox[1])
        menuVBox.addLayout(self.menuBox[2])
        menuVBox.addStretch(10)
        menuVBox.addWidget(btnQuestion)
        menuVBox.addStretch(1)
        widget = QWidget(objectName='menuWidget')
        widget.setStyleSheet('''
        #menuWidget {
            background-color: #333333;
            border-radius: 15px;
        }
        #BtnHome {
            background-color: #333333;
            border-radius: 10px;
            color: #AAAAAA;
            text-align: left;
            font-size: 24px;
            font-weight: bold;
            border: none;
        }
        #BtnHome:hover {
            background-color: #444444;
        }
        #BtnClassification {
            background-color: #333333;
            border-radius: 10px;
            color: #AAAAAA;
            text-align: left;
            font-size: 24px;
            font-weight: bold;
            border: none;
        }
        #BtnClassification:hover {
            background-color: #444444;
        }
        #BtnObjectDetection {
            background-color: #333333;
            border-radius: 10px;
            color: #AAAAAA;
            text-align: left;
            font-size: 24px;
            font-weight: bold;
            border: none;
        }
        #BtnObjectDetection:hover {
            background-color: #444444;
        }
        #BtnQuestion {
            background-color: #333333;
            border-radius: 30px;
            color: #AAAAAA;
            text-align: left;
            font-size: 50px;
            font-weight: bold;
            border: none;
        }
        #BtnQuestion:hover {
            background-color: #444444;
        }
        ''')
        widget.setLayout(menuVBox)
        layout.addWidget(widget)

        # 내용 위젯
        self.contentVBox = QVBoxLayout()
        self.contentVBox.setContentsMargins(5, 5, 5, 5)
        boxTitle = QHBoxLayout()
        lblTitle = QLabel()
        lblTitle.setStyleSheet('background-color: #AA2233; border-radius: 10px;')
        lblTitle.setFixedHeight(25)
        minimumTitle = QPushButton('_', objectName='MinimumTitle')
        minimumTitle.setFixedSize(35, 35)
        minimumTitle.setStyleSheet('''
        #MinimumTitle {
            background-color: #333333;
            border-radius: 15px;
            color: #AAAAAA;
            font-size: 20px;
            font-weight: bold;
            border: none;
        }
        #MinimumTitle:hover {
            background-color: #5A0505;
        }
        ''')
        minimumTitle.clicked.connect(lambda: self.showMinimized())
        closeTitle = QPushButton('X', objectName='CloseTitle')
        closeTitle.setFixedSize(35, 35)
        closeTitle.setStyleSheet('''
        #CloseTitle {
            background-color: #333333;
            border-radius: 15px;
            color: #AAAAAA;
            font-size: 20px;
            font-weight: bold;
            border: none;
        }
        #CloseTitle:hover {
            background-color: #5A0505;
        }
        ''')
        closeTitle.clicked.connect(lambda: self.close())
        boxTitle.addWidget(lblTitle)
        boxTitle.addWidget(minimumTitle)
        boxTitle.addWidget(closeTitle)
        self.status = QLabel('  안녕하세요 :)')
        self.status.setStyleSheet('background-color: #BDC030; border-radius: 10px; font-weight: bold;')
        self.status.setFixedHeight(20)
        self.contentVBox.addLayout(boxTitle)
        self.contentVBox.addStretch(1)
        self.contentVBox.addStretch(1)
        self.contentVBox.addWidget(self.status)
        widget = QWidget(objectName='contentWidget')
        widget.setStyleSheet('#contentWidget{ background-color: #AAAAAA; border-radius: 15px; }')
        widget.setLayout(self.contentVBox)
        layout.addWidget(widget)

        # 레이아웃 적용
        layout.setStretch(1, 1)
        self.mainWidget.setLayout(layout)

        # 클릭
        self.menuBtn[0].click()

    def menuClick(self):
        for i, (btn, lbl) in enumerate(zip(self.menuBtn, self.menuLbl)):
            if btn == self.sender():
                lbl.setStyleSheet('background-color: #882233')
                self.insert_start_widget(self.widget_info[i][0])
                self.debug(self.widget_info[i][2])
            else:
                lbl.setStyleSheet('background-color: #333333')

    def titleClick(self):
        pass

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() != Qt.LeftButton:
            return

        self.drag_flag = True
        self.drag_pos = e.globalPos() - self.pos()
        self.setCursor(QCursor(Qt.ClosedHandCursor))
        e.accept()

    def mouseMoveEvent(self, e: QMouseEvent):
        if not self.drag_flag:
            return

        self.move(e.globalPos() - self.drag_pos)
        e.accept()

    def mouseReleaseEvent(self, e: QMouseEvent):
        self.drag_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
