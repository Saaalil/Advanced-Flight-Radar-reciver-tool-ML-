import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(325, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 50, 191, 41))
        self.pushButton.setObjectName("pushButton")
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 300, 191, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(50, 125, 261, 31))
        self.checkBox_2.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_2.setObjectName("checkBox_2")
        
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(50, 175, 261, 31))
        self.checkBox_3.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_3.setObjectName("checkBox_3")
        
        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(50, 225, 261, 31))
        self.checkBox_5.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_5.setText("Conventional Clustering")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Clustering App"))
        self.pushButton.setText(_translate("MainWindow", "Select CSV File"))
        self.pushButton_2.setText(_translate("MainWindow", "Run Clustering"))

        self.checkBox_2.setText(_translate("MainWindow", "K-Means Clustering"))
        self.checkBox_3.setText(_translate("MainWindow", "DBSCAN Clustering"))
        self.checkBox_5.setText(_translate("MainWindow", "Conventional Clustering"))

class ClusteringApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.file_path = None
        self.pushButton.clicked.connect(self.selectFile)
        self.pushButton_2.clicked.connect(self.generateOutput)

    def selectFile(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv)", options=options)
        if file_name:
            self.file_path = file_name

    def generateOutput(self):
        try:
            if not self.file_path:
                raise ValueError("No CSV file selected.")
            
            if self.checkBox_2.isChecked():  # K-Means
                n_clusters_angle, _ = QtWidgets.QInputDialog.getInt(None, "Enter Number of Clusters for K-Means (Angle)", "Number of Clusters (Angle):", 3, 1)
                n_clusters_frequency, _ = QtWidgets.QInputDialog.getInt(None, "Enter Number of Clusters for K-Means (Frequency)", "Number of Clusters (Frequency):", 3, 1)
                save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save K-Means Output CSV", "", "CSV Files (*.csv)")
                if save_path:
                    self.kmeans_clustering(self.file_path, n_clusters_angle, n_clusters_frequency, save_path)
            
            if self.checkBox_3.isChecked():  # DBSCAN
                eps_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN (Angle)", "Eps (Angle):", 0.5, 0.01)
                min_samples_angle, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN (Angle)", "Min Samples (Angle):", 5, 1)
                eps_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN (Frequency)", "Eps (Frequency):", 0.5, 0.01)
                min_samples_frequency, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN (Frequency)", "Min Samples (Frequency):", 5, 1)
                save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save DBSCAN Output CSV", "", "CSV Files (*.csv)")
                if save_path:
                    self.dbscan_clustering(self.file_path, eps_angle, min_samples_angle, eps_frequency, min_samples_frequency, save_path)

            if self.checkBox_5.isChecked():  # Centroid Clustering
                threshold_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Angle Threshold for Centroid Clustering", "Angle Threshold:", 5.0, 0.1)
                threshold_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Frequency Threshold for Centroid Clustering", "Frequency Threshold:", 100.0, 1.0)
                save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Centroid Clustering Output CSV", "", "CSV Files (*.csv)")
                if save_path:
                    self.centroid_clustering(self.file_path, threshold_angle, threshold_frequency, save_path)
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred: {str(e)}")

    def kmeans_clustering(self, file_path, n_clusters_angle, n_clusters_frequency, save_path):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Check if 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns exist
            if 'Frequency (Hz)' not in df.columns or 'Angle of Arrival (degrees)' not in df.columns:
                raise ValueError("CSV file must contain 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns.")

            # Perform K-Means clustering on Angle of Arrival
            kmeans_angle = KMeans(n_clusters=n_clusters_angle, random_state=42)
            df['Cluster_Angle'] = kmeans_angle.fit_predict(df[['Angle of Arrival (degrees)']])
            
            # Get centroids for Angle of Arrival clusters
            centroids_angle = kmeans_angle.cluster_centers_
            print("Centroids (Angle):", centroids_angle)

            # Perform K-Means clustering on Frequency for each cluster from Angle clustering
            clustered_dfs = []
            for cluster_angle in range(n_clusters_angle):
                mask = df['Cluster_Angle'] == cluster_angle
                X_cluster = df.loc[mask, ['Frequency (Hz)']]

                if len(X_cluster) == 0:
                    continue

                kmeans_frequency = KMeans(n_clusters=n_clusters_frequency, random_state=42)
                df_cluster = df.loc[mask].copy()
                df_cluster['Cluster_Frequency'] = kmeans_frequency.fit_predict(X_cluster)

                # Get centroids for Frequency clusters within each Angle cluster
                centroids_frequency = kmeans_frequency.cluster_centers_
                print(f"Centroids (Frequency) for Angle Cluster {cluster_angle + 1}:", centroids_frequency)

                # Assign centroids to df_cluster
                if centroids_angle.shape[0] > cluster_angle:
                    df_cluster['Centroid_Angle'] = centroids_angle[cluster_angle, 0]  # Angle centroid
                else:
                    df_cluster['Centroid_Angle'] = np.nan

                for freq_cluster in range(n_clusters_frequency):
                    freq_mask = df_cluster['Cluster_Frequency'] == freq_cluster
                    if centroids_frequency.shape[0] > freq_cluster:
                        df_cluster.loc[freq_mask, 'Centroid_Frequency'] = centroids_frequency[freq_cluster, 0]  # Frequency centroid
                    else:
                        df_cluster.loc[freq_mask, 'Centroid_Frequency'] = np.nan

                clustered_dfs.append(df_cluster)

            # Concatenate all clustered dataframes
            clustered_df = pd.concat(clustered_dfs)

            # Save results to CSV
            clustered_df.to_csv(save_path, index=False)

            # Plotting - Angle of Arrival
            plt.figure(figsize=(10, 8))
            plt.scatter(df['Angle of Arrival (degrees)'], df['Frequency (Hz)'], c=df['Cluster_Angle'], cmap='viridis', label='Data points')
            plt.scatter(centroids_angle[:, 0], [df['Frequency (Hz)'].mean()] * n_clusters_angle, s=200, c='red', label='Angle Centroids')
            plt.title('K-Means Clustering - Angle of Arrival')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel('Frequency (Hz)')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plotting - Frequency within each Angle cluster
            plt.figure(figsize=(10, 8))
            for cluster_angle in range(n_clusters_angle):
                mask = df['Cluster_Angle'] == cluster_angle
                plt.scatter(df.loc[mask, 'Angle of Arrival (degrees)'], df.loc[mask, 'Frequency (Hz)'], label=f'Angle Cluster {cluster_angle + 1}')
            plt.title('K-Means Clustering - Frequency within Angle Clusters')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel('Frequency (Hz)')
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during K-Means clustering: {str(e)}")

    def dbscan_clustering(self, file_path, eps_angle, min_samples_angle, eps_frequency, min_samples_frequency, save_path):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Check if 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns exist
            if 'Frequency (Hz)' not in df.columns or 'Angle of Arrival (degrees)' not in df.columns:
                raise ValueError("CSV file must contain 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns.")

            # Perform DBSCAN clustering on Angle of Arrival
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[['Angle of Arrival (degrees)']])
            dbscan_angle = DBSCAN(eps=eps_angle, min_samples=min_samples_angle)
            df['Cluster_Angle'] = dbscan_angle.fit_predict(X_scaled)
            
            # Perform DBSCAN clustering on Frequency for each cluster from Angle clustering
            clustered_dfs = []
            for cluster_angle in np.unique(df['Cluster_Angle']):
                if cluster_angle == -1:  # Skip noise points
                    continue
                mask = df['Cluster_Angle'] == cluster_angle
                X_cluster = df.loc[mask, ['Frequency (Hz)']]

                if len(X_cluster) == 0:
                    continue

                X_scaled_freq = scaler.fit_transform(X_cluster)
                dbscan_frequency = DBSCAN(eps=eps_frequency, min_samples=min_samples_frequency)
                df_cluster = df.loc[mask].copy()
                df_cluster['Cluster_Frequency'] = dbscan_frequency.fit_predict(X_scaled_freq)
                
                clustered_dfs.append(df_cluster)

            # Concatenate all clustered dataframes
            clustered_df = pd.concat(clustered_dfs)

            # Save results to CSV
            clustered_df.to_csv(save_path, index=False)

            # Plotting - Angle of Arrival
            plt.figure(figsize=(10, 8))
            plt.scatter(df['Angle of Arrival (degrees)'], df['Frequency (Hz)'], c=df['Cluster_Angle'], cmap='viridis', label='Data points')
            plt.title('DBSCAN Clustering - Angle of Arrival')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel('Frequency (Hz)')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plotting - Frequency within each Angle cluster
            plt.figure(figsize=(10, 8))
            for cluster_angle in np.unique(df['Cluster_Angle']):
                if cluster_angle == -1:
                    continue
                mask = df['Cluster_Angle'] == cluster_angle
                plt.scatter(df.loc[mask, 'Angle of Arrival (degrees)'], df.loc[mask, 'Frequency (Hz)'], label=f'Angle Cluster {cluster_angle}')
            plt.title('DBSCAN Clustering - Frequency within Angle Clusters')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel('Frequency (Hz)')
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during DBSCAN clustering: {str(e)}")

    def centroid_clustering(self, file_path, threshold_angle, threshold_frequency, save_path):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Check if 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns exist
            if 'Frequency (Hz)' not in df.columns or 'Angle of Arrival (degrees)' not in df.columns:
                raise ValueError("CSV file must contain 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns.")

            # Centroid Clustering - Initialize centroids
            centroids_angle = []
            centroids_frequency = []

            for i, row in df.iterrows():
                angle = row['Angle of Arrival (degrees)']
                frequency = row['Frequency (Hz)']

                assigned = False
                for centroid_angle, centroid_frequency in zip(centroids_angle, centroids_frequency):
                    if abs(centroid_angle - angle) <= threshold_angle and abs(centroid_frequency - frequency) <= threshold_frequency:
                        assigned = True
                        break
                
                if not assigned:
                    centroids_angle.append(angle)
                    centroids_frequency.append(frequency)

            df['Cluster'] = -1
            for i, (centroid_angle, centroid_frequency) in enumerate(zip(centroids_angle, centroids_frequency)):
                mask = (abs(df['Angle of Arrival (degrees)'] - centroid_angle) <= threshold_angle) & (abs(df['Frequency (Hz)'] - centroid_frequency) <= threshold_frequency)
                df.loc[mask, 'Cluster'] = i

            # Save results to CSV
            df.to_csv(save_path, index=False)

            # Plotting
            plt.figure(figsize=(10, 8))
            plt.scatter(df['Angle of Arrival (degrees)'], df['Frequency (Hz)'], c=df['Cluster'], cmap='viridis', label='Data points')
            plt.scatter(centroids_angle, centroids_frequency, s=200, c='red', label='Centroids')
            plt.title('Centroid Clustering')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel('Frequency (Hz)')
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during centroid clustering: {str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ClusteringApp()
    MainWindow.show()
    sys.exit(app.exec_())
