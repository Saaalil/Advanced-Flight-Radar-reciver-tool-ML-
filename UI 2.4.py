from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(946, 539)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 90, 281, 51))
        self.pushButton.setStyleSheet("font: 87 14pt \"Outfit Black\";")
        self.pushButton.setObjectName("pushButton")

        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(560, 170, 261, 31))
        self.checkBox_2.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_2.setObjectName("checkBox_2")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(40, 210, 281, 31))
        self.comboBox.setStyleSheet("font: 87 10pt \"Outfit Black\";")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 160, 291, 31))
        self.label.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.label.setObjectName("label")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 440, 281, 51))
        self.pushButton_2.setStyleSheet("font: 87 14pt \"Outfit Black\";")
        self.pushButton_2.setObjectName("pushButton_2")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(360, 10, 231, 41))
        self.label_2.setStyleSheet("font: 87 26pt \"Outfit Black\";\n"
                                    "")
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(450, 100, 501, 31))
        self.label_3.setStyleSheet("font: 87 14pt \"Outfit Black\";")
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(560, 140, 421, 31))
        self.label_4.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.label_4.setObjectName("label_4")

        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(560, 200, 261, 31))
        self.checkBox_3.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_3.setObjectName("checkBox_3")

        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(560, 230, 261, 31))
        self.checkBox_4.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_4.setObjectName("checkBox_4")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(570, 440, 281, 51))
        self.pushButton_3.setStyleSheet("font: 87 14pt \"Outfit Black\";")
        self.pushButton_3.setObjectName("pushButton_3")

        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(560, 260, 261, 31))
        self.checkBox_5.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_5.setObjectName("checkBox_5")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(39, 259, 281, 171))
        self.widget.setObjectName("widget")

        self.addButton = QtWidgets.QPushButton(self.centralwidget)
        self.addButton.setGeometry(QtCore.QRect(330, 210, 101, 31))
        self.addButton.setObjectName("addButton")
        self.addButton.setText("Go")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 946, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.comboBox.currentIndexChanged.connect(self.handle_algorithm_selection)
        self.addButton.clicked.connect(self.handle_add_spinbox)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Select Input .csv File"))
        self.checkBox_2.setText(_translate("MainWindow", " K-Means"))
        self.comboBox.setItemText(0, _translate("MainWindow", "K - Means"))
        self.comboBox.setItemText(1, _translate("MainWindow", "DB Scan"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Hierarchical Clustering"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Conventional Method"))
        self.label.setText(_translate("MainWindow", "Select  Algorithm for Clustering:"))
        self.pushButton_2.setText(_translate("MainWindow", "Generate Output .csv File"))
        self.label_2.setText(_translate("MainWindow", "THRESHOLD"))
        self.label_3.setText(_translate("MainWindow", "Graphical Comparison For Different Algorithms:"))
        self.label_4.setText(_translate("MainWindow", "Select Algorithms for comparison:"))
        self.checkBox_3.setText(_translate("MainWindow", "DB Scan"))
        self.checkBox_4.setText(_translate("MainWindow", "Hierarchical Clustering"))
        self.pushButton_3.setText(_translate("MainWindow", "Compare"))
        self.checkBox_5.setText(_translate("MainWindow", "Conventional Method"))

    def handle_algorithm_selection(self, index):
        # Clear previous widgets in the widget area
        for i in reversed(range(self.widget.layout().count())):
            widget = self.widget.layout().itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Depending on the selected index, add new widgets
        if index == 0:  # K-Means selected
            # Example: Adding a QSpinBox for K-Means
            spin_box = QtWidgets.QSpinBox(self.widget)
            spin_box.setMinimum(1)  # Set minimum value
            spin_box.setMaximum(100)  # Set maximum value
            spin_box.setSuffix(" clusters")  # Set suffix text
            self.widget.layout().addWidget(spin_box)

        elif index == 1:  # DB Scan selected
            # Example: Adding a QSpinBox for DB Scan
            spin_box = QtWidgets.QSpinBox(self.widget)
            spin_box.setMinimum(1)  # Set minimum value
            spin_box.setMaximum(100)  # Set maximum value
            spin_box.setSuffix(" DB Scan param")  # Set suffix text
            self.widget.layout().addWidget(spin_box)

        elif index == 2:  # Hierarchical Clustering selected
            # Example: Adding a QSpinBox for Hierarchical Clustering
            spin_box = QtWidgets.QSpinBox(self.widget)
            spin_box.setMinimum(1)  # Set minimum value
            spin_box.setMaximum(100)  # Set maximum value
            spin_box.setSuffix(" clusters")  # Set suffix text
            self.widget.layout().addWidget(spin_box)

        elif index == 3:  # Conventional Method selected
            # Example: Adding a QSpinBox for Conventional Method
            spin_box = QtWidgets.QSpinBox(self.widget)
            spin_box.setMinimum(1)  # Set minimum value
            spin_box.setMaximum(100)  # Set maximum value
            spin_box.setSuffix(" param")  # Set suffix text
            self.widget.layout().addWidget(spin_box)

    def handle_add_spinbox(self):
        # Automatically select K-Means in the comboBox
        self.comboBox.setCurrentIndex(0)

        # Call handle_algorithm_selection to add QSpinBox
        self.handle_algorithm_selection(0)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
