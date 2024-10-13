import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox, QMessageBox, QFileDialog, QProgressBar

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
        self.pushButton.clicked.connect(self.select_input_csv)

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
        self.comboBox.currentIndexChanged.connect(self.handle_algorithm_selection)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 160, 291, 31))
        self.label.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.label.setObjectName("label")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 440, 281, 51))
        self.pushButton_2.setStyleSheet("font: 87 14pt \"Outfit Black\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.generate_output_csv)

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
        self.pushButton_3.clicked.connect(self.compare_algorithms)

        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(560, 260, 261, 31))
        self.checkBox_5.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_5.setObjectName("checkBox_5")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(39, 259, 281, 171))
        self.widget.setObjectName("widget")

        # Initialize a QVBoxLayout for self.widget
        self.widget_layout = QtWidgets.QVBoxLayout(self.widget)
        self.widget.setLayout(self.widget_layout)
        
        self.progress_bar = QtWidgets.QProgressBar(MainWindow)
        self.progress_bar.setGeometry(30, 500, 400, 20)  # Adjust position and size as needed
        self.progress_bar.setVisible(False)

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
        
        self.pushButton_2.clicked.connect(self.perform_dbscan_clustering)


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

    def show_progress_bar(self):
        self.progress_bar.setVisible(True)

    def hide_progress_bar(self):
        self.progress_bar.setVisible(False)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        QtWidgets.QApplication.processEvents()

    def handle_algorithm_selection(self, index):
        # Clear previous widgets in the widget area
        for i in reversed(range(self.widget_layout.count())):
            widget = self.widget_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Depending on the selected index, add new widgets
        if index == 0:  # K-Means selected
            label = QtWidgets.QLabel("Enter the number of clusters : ")
            label.setStyleSheet("font: 87 12pt \"Outfit Black\";")
            self.widget_layout.addWidget(label)

            spin_box = QtWidgets.QSpinBox()
            spin_box.setMinimum(1)  # Set minimum value
            spin_box.setMaximum(100)  # Set maximum value
            self.widget_layout.addWidget(spin_box)
            self.kmeans_spin_box = spin_box

        elif index == 1:  # DB Scan selected
            label_eps = QtWidgets.QLabel("Enter EPS: : ")
            label_eps.setStyleSheet("font: 87 12pt \"Outfit Black\";")
            self.widget_layout.addWidget(label_eps)

            spin_box_eps = QDoubleSpinBox()
            spin_box_eps.setMinimum(0.1)  # Set minimum value
            spin_box_eps.setMaximum(100)  # Set maximum value
            self.widget_layout.addWidget(spin_box_eps)
            self.dbscan_spin_box_eps = spin_box_eps

            label_sample = QtWidgets.QLabel("Enter Sample Size: ")
            label_sample.setStyleSheet("font: 87 12pt \"Outfit Black\";")
            self.widget_layout.addWidget(label_sample)

            spin_box_sample = QtWidgets.QSpinBox()
            spin_box_sample.setMinimum(1)  # Set minimum value
            spin_box_sample.setMaximum(100)  # Set maximum value
            self.widget_layout.addWidget(spin_box_sample)
            self.dbscan_spin_box_sample = spin_box_sample

        elif index == 2:  # Hierarchical Clustering selected
            label = QtWidgets.QLabel("Enter the number of clusters : ")
            label.setStyleSheet("font: 87 12pt \"Outfit Black\";")
            self.widget_layout.addWidget(label)

            spin_box = QtWidgets.QSpinBox()
            spin_box.setMinimum(1)  # Set minimum value
            spin_box.setMaximum(100)  # Set maximum value
            self.widget_layout.addWidget(spin_box)
            self.hierarchical_spin_box = spin_box

        elif index == 3:  # Conventional Method selected
            label_freq = QtWidgets.QLabel("Threshold Frequency : ")
            label_freq.setStyleSheet("font: 87 12pt \"Outfit Black\";")
            self.widget_layout.addWidget(label_freq)

            spin_box_freq = QDoubleSpinBox()
            spin_box_freq.setMinimum(0.1)  # Set minimum value
            spin_box_freq.setMaximum(99999999)  # Set maximum value
            self.widget_layout.addWidget(spin_box_freq)
            self.conventional_spin_box_freq = spin_box_freq

            label_aoa = QtWidgets.QLabel("Threshold AoA : ")
            label_aoa.setStyleSheet("font: 87 12pt \"Outfit Black\";")
            self.widget_layout.addWidget(label_aoa)

            spin_box_aoa = QtWidgets.QSpinBox()
            spin_box_aoa.setMinimum(-360)  # Set minimum value
            spin_box_aoa.setMaximum(360)  # Set maximum value
            self.widget_layout.addWidget(spin_box_aoa)
            self.conventional_spin_box_aoa = spin_box_aoa

    def select_input_csv(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Input CSV File", "", "CSV Files (*.csv)", options=options)
        if file_name:
            try:
                self.input_csv_path = file_name
                QMessageBox.information(None, "File Selected", f"Selected CSV file: {file_name}", QMessageBox.Ok)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error selecting CSV file:\n{str(e)}", QMessageBox.Ok)

    def generate_output_csv(self):
        if hasattr(self, 'input_csv_path'):
            try:
                if self.comboBox.currentIndex() == 0 and hasattr(self, 'kmeans_spin_box'):
                    num_clusters = self.kmeans_spin_box.value()
                    output_file = self.perform_kmeans_clustering(self.input_csv_path, num_clusters)
                    if output_file:
                        self.save_output_csv(output_file)
                elif self.comboBox.currentIndex() == 1 and hasattr(self, 'dbscan_spin_box_eps') and hasattr(self, 'dbscan_spin_box_sample'):
                    eps = self.dbscan_spin_box_eps.value()
                    sample_size = self.dbscan_spin_box_sample.value()
                    output_file = self.perform_dbscan_clustering(self.input_csv_path, eps, sample_size)
                    if output_file:
                        self.save_output_csv(output_file)
                elif self.comboBox.currentIndex() == 2 and hasattr(self, 'hierarchical_spin_box'):
                    num_clusters = self.hierarchical_spin_box.value()
                    output_file = self.perform_hierarchical_clustering(self.input_csv_path, num_clusters)
                    if output_file:
                        self.save_output_csv(output_file)
                elif self.comboBox.currentIndex() == 3 and hasattr(self, 'conventional_spin_box_freq') and hasattr(self, 'conventional_spin_box_aoa'):
                    threshold_freq = self.conventional_spin_box_freq.value()
                    threshold_aoa = self.conventional_spin_box_aoa.value()
                    output_file = self.perform_conventional_method(self.input_csv_path, threshold_freq, threshold_aoa)
                    if output_file:
                        self.save_output_csv(output_file)
            except PermissionError:
                QMessageBox.critical(None, "Permission Error", "Permission denied. Check if you have write access to the selected directory.", QMessageBox.Ok)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error generating CSV file:\n{str(e)}", QMessageBox.Ok)

    def perform_kmeans_clustering(self, data_path, num_clusters):
        try:
            data = pd.read_csv(data_path)
            X = data[['Frequency (Hz)', 'Angle of Arrival (degrees)']]
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            data['Cluster'] = kmeans.fit_predict(X)
            
            # Calculate centroids
            centroids = kmeans.cluster_centers_
            
            # Create a single row DataFrame for centroids
            centroid_columns = []
            for i in range(num_clusters):
                centroid_columns.extend([f'Cluster_{i+1}_Centroid_Frequency (Hz)', f'Cluster_{i+1}_Centroid_Angle of Arrival (degrees)'])
            
            centroid_data = pd.DataFrame(columns=centroid_columns)
            for i in range(num_clusters):
                centroid_data.loc[0, f'Cluster_{i+1}_Centroid_Frequency (Hz)'] = centroids[i][0]
                centroid_data.loc[0, f'Cluster_{i+1}_Centroid_Angle of Arrival (degrees)'] = centroids[i][1]
            
            # Merge centroid data with the original data
            merged_data = pd.concat([data, centroid_data], axis=1)
            
            output_file = 'output_kmeans.csv'
            merged_data.to_csv(output_file, index=False)
            
            QMessageBox.information(None, "K-Means Clustering", f"K-Means clustering completed. Output saved to {output_file}", QMessageBox.Ok)
            return output_file
        
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error performing K-Means clustering:\n{str(e)}", QMessageBox.Ok)

    def perform_dbscan_clustering(self):
        try:
            eps = self.dbscan_spin_box_eps.value()
            min_samples = self.dbscan_spin_box_sample.value()

            data = pd.read_csv(self.input_csv_path)
            X = data[['Frequency (Hz)', 'Angle of Arrival (degrees)']]

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)

            data['Cluster'] = clusters

            # Save centroids data to a CSV file
            centroids = pd.DataFrame(dbscan.components_, columns=X.columns)
            centroids.to_csv('centroids_dbscan.csv', index=False)

            output_file = 'output_dbscan.csv'
            data.to_csv(output_file, index=False)

            QMessageBox.information(None, "Clustering Complete", f"DBSCAN clustering completed. Output saved to {output_file}")

            return output_file

        except Exception as e:
            QMessageBox.warning(None, "Error", f"Error performing DBSCAN clustering: {str(e)}")



    def perform_hierarchical_clustering(self, data_path, num_clusters):
        try:
            data = pd.read_csv(data_path)
            X = data[['Frequency (Hz)', 'Angle of Arrival (degrees)']]

            hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
            data['Cluster'] = hierarchical.fit_predict(X)

            # Generate linkage matrix and plot dendrogram
            linkage_matrix = hierarchy.linkage(X, method='ward')  # You can choose other linkage methods as well
            plt.figure(figsize=(10, 6))
            dendrogram = hierarchy.dendrogram(linkage_matrix, truncate_mode='level', p=num_clusters)  # Adjust 'p' as needed
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Sample Index')
            plt.ylabel('Distance')
            plt.tight_layout()

            # Prompt user to select save location for dendrogram
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            save_dendrogram_file, _ = QFileDialog.getSaveFileName(None, "Save Dendrogram Image", "", "PNG Files (*.png)", options=options)
            if save_dendrogram_file:
                plt.savefig(save_dendrogram_file)
                QMessageBox.information(None, "Dendrogram Saved", f"Dendrogram saved to {save_dendrogram_file}", QMessageBox.Ok)
            plt.show()

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error performing Hierarchical clustering:\n{str(e)}", QMessageBox.Ok)

    def perform_conventional_method(self, data_path, threshold_freq, threshold_aoa):
        try:
            data = pd.read_csv(data_path)
            filtered_data = data[(data['Frequency (Hz)'] > threshold_freq) & (data['Angle of Arrival (degrees)'] < threshold_aoa)]
            output_file = 'output_conventional.csv'
            filtered_data.to_csv(output_file, index=False)
            QMessageBox.information(None, "Conventional Method", f"Conventional method completed. Output saved to {output_file}", QMessageBox.Ok)
            return output_file
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error performing Conventional method:\n{str(e)}", QMessageBox.Ok)

    def compare_algorithms(self):
        try:
            if hasattr(self, 'input_csv_path'):
                data = pd.read_csv(self.input_csv_path)

                if self.checkBox_2.isChecked() and hasattr(self, 'kmeans_spin_box'):
                    num_clusters = self.kmeans_spin_box.value()
                    kmeans_output = self.perform_kmeans_clustering(self.input_csv_path, num_clusters)
                    self.plot_graph(data, kmeans_output, 'K-Means Clustering')

                if self.checkBox_3.isChecked() and hasattr(self, 'dbscan_spin_box_eps') and hasattr(self, 'dbscan_spin_box_sample'):
                    eps = self.dbscan_spin_box_eps.value()
                    sample_size = self.dbscan_spin_box_sample.value()
                    dbscan_output = self.perform_dbscan_clustering(self.input_csv_path, eps, sample_size)
                    self.plot_graph(data, dbscan_output, 'DB Scan Clustering')

                if self.checkBox_4.isChecked() and hasattr(self, 'hierarchical_spin_box'):
                    num_clusters = self.hierarchical_spin_box.value()
                    hierarchical_output = self.perform_hierarchical_clustering(self.input_csv_path, num_clusters)
                    self.plot_graph(data, hierarchical_output, 'Hierarchical Clustering')

                if self.checkBox_5.isChecked() and hasattr(self, 'conventional_spin_box_freq') and hasattr(self, 'conventional_spin_box_aoa'):
                    threshold_freq = self.conventional_spin_box_freq.value()
                    threshold_aoa = self.conventional_spin_box_aoa.value()
                    conventional_output = self.perform_conventional_method(self.input_csv_path, threshold_freq, threshold_aoa)
                    self.plot_graph(data, conventional_output, 'Conventional Method')
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error comparing algorithms:\n{str(e)}", QMessageBox.Ok)

    import matplotlib.pyplot as plt

    def plot_graph(self, data, title):
        try:
            plt.figure(figsize=(8, 6))
            plt.title(title)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Angle of Arrival (degrees)')
            plt.scatter(data['Frequency (Hz)'], data['Angle of Arrival (degrees)'], c=data['Cluster'], cmap='viridis')
            plt.show()

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error plotting graph:\n{str(e)}", QMessageBox.Ok)


    def save_output_csv(self, output_file):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            save_file_name, _ = QFileDialog.getSaveFileName(None, "Save Output CSV File", "", "CSV Files (*.csv)", options=options)
            if save_file_name:
                import shutil
                shutil.copy(output_file, save_file_name)
                QMessageBox.information(None, "CSV File Saved", f"CSV file saved to {save_file_name}", QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error saving CSV file:\n{str(e)}", QMessageBox.Ok)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
