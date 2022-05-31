import sys
# 필요한 모둘을 불러옵니다. 기본적인 UI 구성요소를 제공하는 위젯 (클래스)들은 PyQt5.QtWidgets 모듈에 있음
# QIcon 사용하기 위해서 모듈 import
from PyQt5.QtGui import QIcon
import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel,  QAction, qApp, QFileDialog


# QtCore 모듈의 QCoreApplication 클래스를 불러옵니다.
from PyQt5.QtCore import QCoreApplication


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 아이콘 추가
        self.setWindowIcon(QIcon("./web.png"))
        # 프레임 만들기
        """
        setWindowTitle() 메서드는 타이틀바에 나타나는 창의 제목을 설정합니다.
        move() 메서드는 위젯을 스크린의 x=300px, y=300px의 위치로 이동시킵니다.
        resize() 메서드는 위젯의 크기를 너비 400px, 높이 200px로 조절합니다.    
        show() 메서드는 위젯을 스크린에 보여줍니다.
        """
        self.setWindowTitle('My First Application')
        self.move(300, 300)
        self.resize(400, 200)

        # 창 닫기 버튼 추가
        """
        푸시버튼을 하나 만듭니다.
        이 버튼 (btn)은 QPushButton 클래스의 인스턴스입니다.
        생성자 (QPushButton())의 첫 번째 파라미터에는 버튼에 표시될 텍스트를 입력하고, 두 번째 파라미터에는 버튼이 위치할 부모 위젯을 입력합니다.
        푸시버튼 위젯에 대한 자세한 설명은 QPushButton 페이지를 참고하세요.
        """
        btn = QPushButton('Quit', self)
        btn.move(0,20)
        btn.resize(btn.sizeHint())
        btn.clicked.connect(QCoreApplication.instance().quit)

        # 상태바 테스트 위한 코드
        self.statusBar().showMessage('준비중...')
        # 상태창 테스트를 위한 label 추가
        self.label = QLabel("00000000000", self)
        self.label.move(40, 40)
        # 상태창 테스트를 위한 Start 버튼 추가
        self.pb = QPushButton("Start", self)
        self.pb.clicked.connect(self.count_number)
        self.pb.move(150, 40)

        # 메뉴바 만들기
        exitAction = QAction('&Exit', self)
        # 단축키
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        # 추가
        loadfile = QAction('laod File ...', self)
        savefile = QAction('save File ...', self)
        loadfile.triggered.connect(self.add_open)
        savefile.triggered.connect(self.add_save)
        fileMenu.addAction(exitAction)
        fileMenu.addAction(loadfile)
        fileMenu.addAction(savefile)

        self.show()

    def count_number(self) :
        """
        상태바 테스트 함수
        """
        # 상태바 생성 코드

        self.statusBar().showMessage('작업중...')
        # PyQt5를 쓰면서 데이터가 계속 업데이트 되지 않을 때 반드시 repaint() 부분을 실행해주도록 합니다.
        self.statusBar().repaint()
        for i in range(1, 100000) :
            print(i)
            self.label.setText(str(i))
            self.label.repaint()

        self.statusBar().showMessage("준비중...")

    def add_open(self):
        FileOpen = QFileDialog.getOpenFileName(self, 'Open file', './')
        print(FileOpen)

    def add_save(self):
        FileSave = QFileDialog.getSaveFileName(self, 'Save file', './')
        print(FileSave)


if __name__ == '__main__':
   # 모든 PyQt5 어플리케이션은 어플리케이션 객체를 생성해야 합니다.
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())