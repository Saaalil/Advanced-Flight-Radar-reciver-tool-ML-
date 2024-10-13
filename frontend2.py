import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(365, 539)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Select Input File Button
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(40, 60, 281, 51))
        self.pushButton.setStyleSheet("font: 87 14pt \"Outfit Black\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.selectFile)  # Connect button to selectFile method
        
        # Algorithm Selection Label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 120, 291, 31))
        self.label.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.label.setObjectName("label")
        
        # K-Means Checkbox
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(50, 150, 261, 31))
        self.checkBox_2.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_2.setObjectName("checkBox_2")
        
        # DBSCAN Checkbox
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(50, 180, 261, 31))
        self.checkBox_3.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_3.setObjectName("checkBox_3")
        
        # Threshold Label
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 10, 231, 41))
        self.label_2.setStyleSheet("font: 87 26pt \"Outfit Black\";\n"
                                    "")
        self.label_2.setObjectName("label_2")
        
        # Generate Output Button
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 440, 281, 51))
        self.pushButton_2.setStyleSheet("font: 87 14pt \"Outfit Black\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.generateOutput)  # Connect button to generateOutput method
        
        # Central Widget Setup
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Menu Bar Setup
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 365, 21))
        self.menubar.setObjectName("menubar")
        self.menuUser_Guide = QtWidgets.QMenu(self.menubar)
        self.menuUser_Guide.setObjectName("menuUser_Guide")
        MainWindow.setMenuBar(self.menubar)
        
        # Status Bar Setup
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        # Actions and Slots Connection
        self.menubar.addAction(self.menuUser_Guide.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Select Input .csv File"))
        self.checkBox_2.setText(_translate("MainWindow", "K-Means"))
        self.label.setText(_translate("MainWindow", "Select Algorithm for Clustering :"))
        self.pushButton_2.setText(_translate("MainWindow", "Generate Output File"))
        self.label_2.setText(_translate("MainWindow", "THRESHOLD"))
        self.checkBox_3.setText(_translate("MainWindow", "DBSCAN"))
        self.menuUser_Guide.setTitle(_translate("MainWindow", "User Guide"))

    def selectFile(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)", options=options)
        if file_path:
            self.file_path = file_path
            QtWidgets.QMessageBox.information(None, "File Selected", f"Selected file: {file_path}")

    def generateOutput(self):
        try:
            if not hasattr(self, 'file_path'):
                raise ValueError("No CSV file selected.")
            
            if self.checkBox_2.isChecked():
                n_clusters, _ = QtWidgets.QInputDialog.getInt(None, "Enter Number of Clusters for K-Means", "Number of Clusters:", 3, 1)
                self.kmeans_clustering(self.file_path, n_clusters)
            
            if self.checkBox_3.isChecked():
                eps, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN", "Eps:", 0.5, 0.01)
                min_samples, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN", "Min Samples:", 5, 1)
                self.dbscan_clustering(self.file_path, eps, min_samples)
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred: {str(e)}")

    def kmeans_clustering(self, file_path, n_clusters):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns exist
            if 'Frequency (Hz)' not in df.columns or 'Angle of Arrival (degrees)' not in df.columns:
                raise ValueError("CSV file must contain 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns.")
            
            # Extract relevant columns
            X = df[['Frequency (Hz)', 'Angle of Arrival (degrees)']]
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            
            # Add cluster labels to the dataframe
            df['Cluster'] = kmeans.labels_
            
            # Compute median values for each cluster
            median_values = df.groupby('Cluster')[['Frequency (Hz)', 'Angle of Arrival (degrees)']].median()
            median_values.columns = ['Median Frequency (Hz)', 'Median Angle of Arrival (degrees)']
            
            # Calculate and print inertia
            inertia = kmeans.inertia_
            print(f"Inertia: {inertia}")
            
            # Calculate and print silhouette score
            silhouette_avg = silhouette_score(X, kmeans.labels_)
            print(f"Silhouette Score: {silhouette_avg}")
            
            # Add metrics to the median_values DataFrame
            median_values['Inertia'] = inertia
            median_values['Silhouette Score'] = silhouette_avg
            
            # Save median values with metrics to CSV
            output_dir = "kmeans_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            median_output_file_path = os.path.join(output_dir, "median_values_with_metrics.csv")
            median_values.to_csv(median_output_file_path)
            print(f"Median values with metrics saved to {median_output_file_path}")
            
            # Plotting
            plt.figure(figsize=(12, 8))  # Larger figure size for better detail
            
            # Scatter plot of data points
            plt.scatter(df['Frequency (Hz)'], df['Angle of Arrival (degrees)'], c=df['Cluster'], cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5, label='Data Points')
            
            # Scatter plot of centroids with annotations
            centroids = kmeans.cluster_centers_
            for i, centroid in enumerate(centroids):
                plt.scatter(centroid[0], centroid[1], marker='o', s=300, c='red', edgecolors='k', linewidth=1.5, label=f'Centroid {i}')
                plt.text(centroid[0], centroid[1], f' {i}', fontsize=12, ha='center', va='center', color='black')
            
            plt.colorbar(label='Cluster')
            plt.xlabel('Frequency (Hz)', fontsize=14)
            plt.ylabel('Angle of Arrival (degrees)', fontsize=14)
            plt.title(f'K-Means Clustering with {n_clusters} Clusters\nInertia: {inertia:.2f}, Silhouette Score: {silhouette_avg:.2f}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=12, markerscale=0.8)
            plt.tight_layout()
            
            # Save plot as PNG
            output_dir = "kmeans_output"
            plot_output_file_path = os.path.join(output_dir, "kmeans_plot_with_metrics.png")
            plt.savefig(plot_output_file_path)
            print(f"Plot saved as {plot_output_file_path}")
            
            # Show plot
            plt.show()
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during K-Means clustering: {str(e)}")
    
    def dbscan_clustering(self, file_path, eps, min_samples):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns exist
            if 'Frequency (Hz)' not in df.columns or 'Angle of Arrival (degrees)' not in df.columns:
                raise ValueError("CSV file must contain 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns.")
            
            # Extract relevant columns
            X = df[['Frequency (Hz)', 'Angle of Arrival (degrees)']]
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X_scaled)
            
            # Add cluster labels to the dataframe
            df['Cluster'] = dbscan.labels_
            
            # Filter out noise points (cluster label -1) from the DataFrame
            df_clusters_only = df[df['Cluster'] != -1]
            
            # Save only clusters to CSV file
            output_dir = "dbscan_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file_path = os.path.join(output_dir, "dbscan_clusters_only.csv")
            df_clusters_only.to_csv(output_file_path, index=False)
            print(f"Clusters only saved to {output_file_path}")
            
            # Calculate silhouette score
            X_noise_removed = X_scaled[df['Cluster'] != -1]
            labels_noise_removed = dbscan.labels_[df['Cluster'] != -1]
            silhouette_avg = silhouette_score(X_noise_removed, labels_noise_removed)
            print(f"Silhouette Score: {silhouette_avg}")
            
            # Plotting clusters
            plt.figure(figsize=(12, 8))  # Larger figure size for better detail
            
            # Define colors for each cluster
            colors = plt.cm.nipy_spectral(dbscan.labels_.astype(float) / dbscan.labels_.max())
            
            # Scatter plot of data points
            plt.scatter(df['Frequency (Hz)'], df['Angle of Arrival (degrees)'], c=colors, cmap='nipy_spectral', s=50, alpha=0.7, edgecolors='w', linewidth=0.5, label='Data Points')
            
            # Annotate each cluster with a number
            clusters = df['Cluster'].unique()
            for cluster in clusters:
                if cluster == -1:
                    continue  # Skip noise points
                cluster_center = X[df['Cluster'] == cluster].mean(axis=0)
                plt.text(cluster_center[0], cluster_center[1], f' Cluster {cluster}', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            
            plt.colorbar(label='Cluster')
            plt.xlabel('Frequency (Hz)', fontsize=14)
            plt.ylabel('Angle of Arrival (degrees)', fontsize=14)
            plt.title(f'DBSCAN Clustering\nEps={eps}, Min Samples={min_samples}\nSilhouette Score: {silhouette_avg:.2f}', fontsize=16)  # Include silhouette score in title
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=12, markerscale=0.8)
            plt.tight_layout()
            
            # Save plot as PNG
            output_dir = "dbscan_output"
            plot_output_file_path = os.path.join(output_dir, "dbscan_plot.png")
            plt.savefig(plot_output_file_path)
            print(f"Plot saved as {plot_output_file_path}")
            
            # Show plot
            plt.show()
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during DBSCAN clustering: {str(e)}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
