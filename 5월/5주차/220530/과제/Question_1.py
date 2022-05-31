from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(536, 571)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.horizontalLayout.addWidget(self.verticalSlider)
        self.verticalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider_2")
        self.horizontalLayout.addWidget(self.verticalSlider_2)
        self.verticalSlider_3 = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_3.setObjectName("verticalSlider_3")
        self.horizontalLayout.addWidget(self.verticalSlider_3)
        self.verticalSlider_4 = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_4.setObjectName("verticalSlider_4")
        self.horizontalLayout.addWidget(self.verticalSlider_4)
        self.horizontalLayout_3.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.verticalSlider.valueChanged['int'].connect(self.brightness_value)
        self.verticalSlider_2.valueChanged['int'].connect(self.blur_value)
        self.verticalSlider_3.setMinimum(-1)
        self.verticalSlider_3.setMaximum(1)
        self.verticalSlider_3.valueChanged['int'].connect(self.flip_value)
        self.verticalSlider_4.setMinimum(0)
        self.verticalSlider_4.setMaximum(180)
        self.verticalSlider_4.valueChanged['int'].connect(self.rotation_value)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton.clicked.connect(self.savePhoto)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.image = None

        # Added code here
        self.filename = None  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display
        self.brightness_value_now = 0  # Updated brightness value
        self.blur_value_now = 0  # Updated blur value
        self.flip_value_now = 1
        self.verticalSlider_3.setValue(self.flip_value_now)
        self.rotation_value_now = 45
        self.verticalSlider_4.setValue(self.rotation_value_now)

    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        import os
        if os.path.isfile(self.filename) is False:
            return

        np_image = np.fromfile(self.filename, np.uint8)
        self.image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        #self.image = cv2.imread(self.filename)
        self.setPhoto(self.image)
        self.update()

    def setPhoto(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def brightness_value(self, value):
        """ This function will take value from the slider
            for the brightness from 0 to 99
        """
        self.brightness_value_now = value
        print('Brightness: ', value)
        self.update()

    def blur_value(self, value):
        """ This function will take value from the slider
            for the blur from 0 to 99 """
        self.blur_value_now = value
        print('Blur: ', value)
        self.update()

    def flip_value(self, value):
        self.flip_value_now = value
        print('Flip: ', value)
        self.update()

    def rotation_value(self, value):
        self.rotation_value_now = value
        print('Rotation: ', value)
        self.update()

    def changeBrightness(self, img, value):
        """ This function will take an image (img) and the brightness
            value. It will perform the brightness change using OpenCv
            and after split, will merge the img and return it.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def changeBlur(self, img, value):
        """ This function will take the img image and blur values as inputs.
            After perform blur operation using opencv function, it returns
            the image img.
        """
        kernel_size = (value + 1, value + 1)  # +1 is to avoid 0
        img = cv2.blur(img, kernel_size)
        return img

    def changeFlip(self, img, value):
        img = cv2.flip(img, value)
        return img

    def changeRotation(self, img, value):
        height, width, channel = img.shape

        matrix = cv2.getRotationMatrix2D((width/2, height/2), value, 1)
        img = cv2.warpAffine(img, matrix, (width, height))
        return img

    def update(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        if self.image is None:
            return
        img = self.changeBrightness(self.image, self.brightness_value_now)
        img = self.changeBlur(img, self.blur_value_now)
        img = self.changeFlip(img, self.flip_value_now)
        img = self.changeRotation(img, self.rotation_value_now)
        self.setPhoto(img)

    def savePhoto(self):
        """ This function will save the image"""
        # here provide the output file name
        # lets say we want to save the output as a time stamp
        # uncomment the two lines below

        # import time
        # filename = 'Snapshot '+str(time.strftime("%Y-%b-%d at %H.%M.%S %p"))+'.png'

        # Or we can give any name such as output.jpg or output.png as well
        # filename = 'Snapshot.png'

        # Or a much better option is to let user decide the location and the extension
        # using a file dialog.

        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]

        from PIL import Image
        Image.fromarray(cv2.cvtColor(self.tmp, cv2.COLOR_BGR2RGB)).save(filename)
        #cv2.imwrite(filename, self.tmp)
        print('Image saved as:', filename)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pyshine photo editor"))
        self.pushButton_2.setText(_translate("MainWindow", "Open"))
        self.pushButton.setText(_translate("MainWindow", "Save"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
