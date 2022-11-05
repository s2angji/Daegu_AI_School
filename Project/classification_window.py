from torch.utils.data import DataLoader
import torch.optim as optim
import threading
import time

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from main_window import *
from classification_ml import *


class ClassificationWidget(QWidget):

    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main

        # 모델 훈련에 사용되는 변수 선언
        self.m_data_transform = None
        self.m_train_data = None
        self.m_test_data = None
        self.m_train_loader = None
        self.m_test_loader = None
        self.m_model = None
        self.m_model_name = None
        self.m_image_size = None
        self.m_optimizer = None
        self.m_lr_scheduler = None

        # 시간이 다소 소요되는 모델 학습을 위한 사용자 정의 시그널
        def start():
            self.main.debug('시작 !!!!!!!!!!!!!!!!!')
            self.debug.setText('시작 !!!!!!!!!!!!!!!!!\n\n')

            self.fig.clear()
            self.canvas.draw()

            self.train_btn.setEnabled(False)
            self.btn_loss_acc.setEnabled(False)
            self.btn_predict.setEnabled(False)
            self.btn_class_acc.setEnabled(False)
            self.resnet_btn.setEnabled(False)
            self.vgg_btn.setEnabled(False)
            self.alexnet_btn.setEnabled(False)
            self.slider.setEnabled(False)

        def progress(value, message):
            self.main.debug('진행율 : {0:3d}'.format(value))

            self.progressBar.setValue(value)
            self.debug.setText(self.debug.toPlainText() + message)
            scroll_bar = self.debug.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())

        def done(b: bool):
            self.main.debug('완료 !!!!!!!!!!!!!!!!!')

            self.train_btn.setEnabled(True)
            self.btn_loss_acc.setEnabled(True)
            self.btn_predict.setEnabled(True)
            self.btn_class_acc.setEnabled(True)
            self.resnet_btn.setEnabled(True)
            self.vgg_btn.setEnabled(True)
            self.alexnet_btn.setEnabled(True)
            self.slider.setEnabled(True)

            # True: 모델 학습 종료 후, Loss 및 Acc 그래프 그리기 위해서 버튼 클릭
            # False: 학습한 모델의 클래스 별 정확도 그래프 그리기
            if b:
                self.btn_loss_acc.click()
            else:
                self.draw_class_acc()

        self.signal = Communicate()
        self.signal.start_signal.connect(start)
        self.signal.progress_signal.connect(progress)
        self.signal.done_signal.connect(done)

        self.debug = QTextEdit()
        self.debug.setText(f'Start 버튼으로 학습을 시작해보세요!! ({device} 사용)\n')
        self.debug.setReadOnly(True)
        self.debug.setFixedHeight(170)
        self.debug.verticalScrollBar().setStyleSheet('''
        QScrollBar:vertical {
            border: 2px solid #999999;
            background-color: #333333;
            width: 35px;
            margin: 0px 10px 0px 0px;
        }
        QScrollBar::handle:vertical {
            min-height: 0px;
            background-color: #333333;
        }
        QScrollBar::add-line:vertical {
            height: 0px;
            subcontrol-position: bottom;
            subcontrol-origin: margin;
        }
        QScrollBar::sub-line:vertical {
            height: 0 px;
            subcontrol-position: top;
            subcontrol-origin: margin;
        }''')
        self.debug.setStyleSheet('''
        QTextEdit {
            background-color: #404040;
            color: #999999;
            border-radius: 5px;
            font-family: consolas, "Malgun Gothic", serif;
            font-size: 20px;
            padding: 15px;
        }''')
        self.progressBar = QProgressBar()
        self.progressBar.setLayoutDirection(Qt.LeftToRight)
        self.progressBar.setStyleSheet('''
        QProgressBar {
            background-color: #AA2233;
            color: #333333;
            border-style: none;
            border-bottom-right-radius: 5px;
            border-bottom-left-radius: 5px;
            border-top-right-radius: 5px;
            border-top-left-radius: 5px;
            text-align: center;
            height: 30px;
            font-size: 20px;
            font-weight: bold;
        }
        QProgressBar::chunk {
            border-bottom-right-radius: 5px;
            border-bottom-left-radius: 5px;
            border-top-right-radius: 5px;
            border-top-left-radius: 5px;
            background-color: qlineargradient(
                spread:pad,
                x1:0, y1:0.511364,
                x2:1, y2:0.523,
                stop:0 rgba(189, 192, 48, 255),
                stop:1 rgba(189, 192, 48, 255));
        }
        ''')
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.debug)
        self.vbox.addWidget(self.progressBar)

        hbox = QHBoxLayout()
        self.lcd = QLCDNumber(self)
        self.lcd.setStyleSheet('''
        border: none;
        color: #FFFFFF;
        ''')
        def on_value_changed_slider(n):
            global num_epochs
            num_epochs = n
            self.lcd.display(n)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksAbove)
        self.slider.valueChanged.connect(on_value_changed_slider)
        self.slider.setValue(num_epochs)
        self.slider.setFixedWidth(350)
        self.slider.setStyleSheet('''
        QSlider::groove:horizontal {
            background-color: #D0D0D0;
            border-radius: 3px;
            height: 15px;
            margin: 0px;
            }
        QSlider::handle:horizontal {
            background-color: #393939;
            border-radius: 7px;
            border: 1px solid;
            height: 40px;
            width: 40px;
            margin: -7px 0px;
        }''')
        def on_click_radio_button():
            self.m_model_name = self.sender().text().lower().strip()
        self.resnet_btn = QRadioButton(' Resnet')
        self.resnet_btn.toggled.connect(on_click_radio_button)
        self.resnet_btn.setFixedWidth(100)
        self.resnet_btn.setStyleSheet('font-size: 20px; font-weight: bold;')
        self.vgg_btn = QRadioButton(' Vgg')
        self.vgg_btn.toggled.connect(on_click_radio_button)
        self.vgg_btn.setFixedWidth(70)
        self.vgg_btn.setStyleSheet("QRadioButton::indicator:checked:pressed{ background-color : lightred; }")
        self.vgg_btn.setStyleSheet('font-size: 20px; font-weight: bold;')
        self.alexnet_btn = QRadioButton(' Alexnet')
        self.alexnet_btn.toggled.connect(on_click_radio_button)
        self.alexnet_btn.setFixedWidth(120)
        self.alexnet_btn.setStyleSheet("QRadioButton::indicator:checked:pressed{ background-color : lightred; }")
        self.alexnet_btn.setStyleSheet('font-size: 20px; font-weight: bold;')
        pixmap = QPixmap('./img/train.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.train_btn = QPushButton('  Start')
        self.train_btn.clicked.connect(self.on_click_train_btn)
        self.train_btn.setIcon(icon)
        self.train_btn.setFixedWidth(180)
        self.train_btn.setStyleSheet('''
        QPushButton {
            border-radius: 5px;
            background-color: #333333;
            color: #AAAAAA;
            height: 50px;
            font-size: 20px;
            font-weight: bold;
        }
        QPushButton:hover {
             background-color: #666666;
        }''')  # border-image: url(./img/2.png);
        hbox.addWidget(self.resnet_btn)
        hbox.addWidget(self.vgg_btn)
        hbox.addWidget(self.alexnet_btn)
        hbox.addStretch(4)
        hbox.addWidget(self.lcd)
        hbox.addWidget(self.slider)
        hbox.addStretch(1)
        hbox.addWidget(self.train_btn)
        self.vbox.addLayout(hbox)

        self.vbox.addStretch(1)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet('border-radius: 5px;')
        self.vbox.addWidget(self.canvas)

        self.vbox.addStretch(1)

        hbox = QHBoxLayout()
        pixmap = QPixmap('./img/loss_and_acc.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.btn_loss_acc = QPushButton('  Loss and Acc')
        self.btn_loss_acc.setEnabled(False)
        self.btn_loss_acc.clicked.connect(self.on_click_btn_loss_acc)
        self.btn_loss_acc.setIcon(icon)
        self.btn_loss_acc.setStyleSheet('''
        QPushButton {
            border-radius: 5px;
            background-color: #333333;
            color: #AAAAAA;
            height: 40px;
            font-size: 20px;
            font-weight: bold;
        }
        QPushButton:hover {
             background-color: #666666;
        }''')
        pixmap = QPixmap('./img/predict.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.btn_predict = QPushButton('  Predict')
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.on_click_btn_predict)
        self.btn_predict.setIcon(icon)
        self.btn_predict.setStyleSheet('''
        QPushButton {
            border-radius: 5px;
            background-color: #333333;
            color: #AAAAAA;
            height: 40px;
            font-size: 20px;
            font-weight: bold;
        }
        QPushButton:hover {
             background-color: #666666;
        }''')
        pixmap = QPixmap('./img/class_acc.png')
        icon = QIcon()
        icon.addPixmap(pixmap)
        self.btn_class_acc = QPushButton('  Class Acc')
        self.btn_class_acc.setEnabled(False)
        self.btn_class_acc.clicked.connect(self.on_click_btn_class_acc)
        self.btn_class_acc.setIcon(icon)
        self.btn_class_acc.setStyleSheet('''
        QPushButton {
            border-radius: 5px;
            background-color: #882233;
            color: #AAAAAA;
            height: 40px;
            font-size: 20px;
            font-weight: bold;
        }
        QPushButton:hover {
             background-color: #AA3344;
        }''')
        hbox.addWidget(self.btn_loss_acc)
        hbox.addWidget(self.btn_predict)
        hbox.addWidget(self.btn_class_acc)
        self.vbox.addLayout(hbox)
        self.setLayout(self.vbox)

        self.alexnet_btn.setChecked(True)

    # def visualize(self):
    #     visualize_loss_acc(history=self.hist, optim=f'SGD ; lr={lr}, step_size=4, gamma=0.1')
    #     visualize_predict(model=self.model_ft, data=self.m_test_data)
    #     sgd_lr025_dict_resnet = visualize_class_accuracies(model=self.model_ft, data=self.m_test_data)
    #     save_model(self.model_ft, 'resnet_best.pt')

    def on_click_btn_loss_acc(self):
        self.fig.clear()

        ax = self.fig.add_subplot(121)
        ax.plot(self.hist['train_loss'], label='train_loss')
        ax.plot(self.hist['valid_loss'], label='valid_loss')
        ax.legend()
        ax.set_title('Loss Curves')

        ax = self.fig.add_subplot(122)
        ax.plot(self.hist['train_acc'], label='train_acc')
        ax.plot(self.hist['valid_acc'], label='valid_acc')
        ax.legend()
        ax.set_title('Accuracy Curves')

        self.canvas.draw()

    def on_click_btn_predict(self):
        c = np.random.randint(0, len(self.m_test_data))
        img, label = self.m_test_data[c]

        self.fig.clear()

        model = self.model_ft
        data = self.m_test_data

        with torch.no_grad():
            model.eval()
            # Model outputs log probabilities
            # out = model(img.view(1, 3, 224, 224).cuda())
            out = model(img.view(1, 3, 224, 224).cpu())
            out = torch.exp(out)
            # print(out)

        ax = self.fig.add_subplot(121)
        # ax.imshow(img.numpy().transpose((1, 2, 0)).astype(np.uint8))
        ax.imshow(Image.open(data.all_data[c]).convert('RGB'))
        ax.set_title(self.m_test_data.labels[label])

        ax = self.fig.add_subplot(122)
        ax.barh(data.labels, np.nan_to_num(out.cpu().numpy()[0]))

        self.canvas.draw()

        self.debug.setText(f'{self.debug.toPlainText()}\n\n'
                           + ('(예측 성공) ' if out.cpu().numpy()[0].argmax() == label else '(예측 실패) ')
                           + f'{self.m_test_data.labels[label]}을(를) '
                           + f'{self.m_test_data.labels[out.cpu().numpy()[0].argmax()]}로(으로) 예측했습니다.')
        scroll_bar = self.debug.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def on_click_btn_class_acc(self):
        model = self.model_ft
        data = self.m_test_data

        def run():
            self.main.signal.start_signal.emit()
            self.signal.start_signal.emit()

            self.accuracy_dict = {}
            with torch.no_grad():
                model.eval()
                for i, c in enumerate(data.all_data_dict.keys()):
                    total_count = len(data.all_data_dict[c])
                    correct_count = 0
                    for path in data.all_data_dict[c]:
                        # print(path)
                        im = Image.open(path).convert('RGB')
                        # im.show()
                        im = transforms.ToTensor()(im)
                        im = transforms.Resize((224, 224))(im)
                        # out = model(im.view(1, 3, 224, 224).cuda())
                        out = model(im.view(1, 3, 224, 224).cpu())
                        # print(out)
                        out = torch.exp(out)
                        pred = list(out.cpu().numpy()[0])
                        # print(pred)
                        pred = pred.index(max(pred))
                        # print(pred, data.labels.index(c))

                        if pred == data.labels.index(c):
                            correct_count += 1

                    self.signal.progress_signal.emit(100 / len(data.all_data_dict.keys()) * (i + 1),
                                                     'Accuracy for class {0} : {1:.3f}\n'.format(
                                                         c, correct_count / total_count
                                                     ))
                    self.accuracy_dict[c] = correct_count / total_count
                    time.sleep(2)

            self.main.signal.done_signal.emit(False)
            self.signal.done_signal.emit(False)

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()

    def draw_class_acc(self):
        self.fig.clear()

        ax = self.fig.add_subplot(111)
        ax.barh(list(self.accuracy_dict.keys()), list(self.accuracy_dict.values()))
        ax.set_title('학습한 모델의 종류별 예측 정확도')

        self.canvas.draw()

    def on_click_train_btn(self):
        def run():
            self.main.signal.start_signal.emit()
            self.signal.start_signal.emit()

            self.signal.progress_signal.emit(0, f'0. selected model: {self.m_model_name}, epoch: {num_epochs}\n')

            # 1. augmentation setting
            self.m_data_transform = data_augmentation()
            self.signal.progress_signal.emit(0, '1. augmentation setting\n')

            # 2. data set setting
            self.m_train_data = CustomDataset(data_path=data_path, mode=Mode.train,
                                              transform=self.m_data_transform['train'])
            self.m_test_data = CustomDataset(data_path=data_path, mode=Mode.val,
                                             transform=self.m_data_transform['test'])
            self.signal.progress_signal.emit(0, '2. data set setting\n')
            # self.main.debug(self.m_train_data.labels[self.m_train_data[0][1]])
            # plt.figure(figsize=(10, 10))
            # plt.imshow(transforms.ToPILImage()(self.m_train_data[0][0]))
            # plt.show()

            # 3. data loader setting
            self.m_train_loader = DataLoader(self.m_train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            self.m_test_loader = DataLoader(self.m_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
            self.signal.progress_signal.emit(0, '3. data loader setting\n')

            # 4. model call
            self.signal.progress_signal.emit(0, '4. model call (pretrained)\n')
            self.m_model, self.m_image_size = initialize_model(self.m_model_name, num_classes=nc)

            # 5. 하이퍼파라메타 값 call loss function 호출, optim, lr_scheduler
            # criterion = nn.CrossEntropyLoss().to(device)  #classification_ml.py에 정의 됨
            self.m_optimizer = optim.SGD(self.m_model.parameters(), lr=lr, momentum=0.9)
            self.m_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.m_optimizer, step_size=4, gamma=0.1)
            self.signal.progress_signal.emit(0, '5. 하이퍼파라메타 값 call loss function 호출, optim, lr_scheduler\n')

            # 6. train loop 함수 호출
            # 7. test loop 함수 호출
            self.signal.progress_signal.emit(0, '6. train, val loop 함수 호출\n\n\n\n')
            self.model_ft, self.hist = train_model(
                self.m_model,
                {'train': self.m_train_loader, 'val': self.m_test_loader},
                criterion,
                self.m_optimizer,
                self.m_lr_scheduler,
                num_epochs,
                self.signal.progress_signal
            )

            save_model(self.model_ft, 'resnet_best.pt')

            self.main.signal.done_signal.emit(True)
            self.signal.done_signal.emit(True)

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.set_func_widget(1, ClassificationWidget)
    mainWindow.show()
    sys.exit(app.exec_())
