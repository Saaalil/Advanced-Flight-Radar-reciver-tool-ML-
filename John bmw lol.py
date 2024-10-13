import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


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
                self.kmeans_clustering(self.file_path, n_clusters_angle, n_clusters_frequency)
            
            if self.checkBox_3.isChecked():  # DBSCAN
                eps_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN (Angle)", "Eps (Angle):", 0.5, 0.01)
                min_samples_angle, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN (Angle)", "Min Samples (Angle):", 5, 1)
                eps_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN (Frequency)", "Eps (Frequency):", 0.5, 0.01)
                min_samples_frequency, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN (Frequency)", "Min Samples (Frequency):", 5, 1)
                self.dbscan_clustering(self.file_path, eps_angle, min_samples_angle, eps_frequency, min_samples_frequency)

            if self.checkBox_5.isChecked():  # Centroid Clustering
                threshold_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Angle Threshold for Centroid Clustering", "Angle Threshold:", 5.0, 0.1)
                threshold_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Frequency Threshold for Centroid Clustering", "Frequency Threshold:", 100.0, 1.0)
                centroid_clustering(self.file_path, threshold_angle, threshold_frequency)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred: {str(e)}")

    def kmeans_clustering(self, file_path, n_clusters_angle, n_clusters_frequency):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns exist
            if 'Frequency (Hz)' not in df.columns or 'Angle of Arrival (degrees)' not in df.columns:
                raise ValueError("CSV file must contain 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns.")
            
            # Extract relevant columns
            X = df[['Frequency (Hz)', 'Angle of Arrival (degrees)']]
            
            # Perform K-Means clustering on Angle of Arrival
            kmeans_angle = KMeans(n_clusters=n_clusters_angle, random_state=42)
            df['Cluster_Angle'] = kmeans_angle.fit_predict(X[['Angle of Arrival (degrees)']])

             # Save K-Means results to CSV
            df.to_csv('kmeans_output.csv', index=False)  # Adjust the file path as needed
        

            
            # Plotting - Angle of Arrival
            plt.figure(figsize=(10, 8))
            plt.scatter(df['Angle of Arrival (degrees)'], df['Frequency (Hz)'], c=df['Cluster_Angle'], cmap='viridis', s=50)
            plt.title('K-Means Clustering (Angle of Arrival)')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Cluster')
            plt.grid(True)
            plt.show()
            
            # Initialize list to store clustered dataframes
            clustered_dfs = []
            
            # Perform K-Means clustering on Frequency for each cluster from Angle clustering
            for cluster_angle in range(n_clusters_angle):
                mask = df['Cluster_Angle'] == cluster_angle
                X_cluster = df.loc[mask, ['Frequency (Hz)']]
                
                if len(X_cluster) == 0:
                    continue
                
                kmeans_frequency = KMeans(n_clusters=n_clusters_frequency, random_state=42)
                df_cluster = df.loc[mask].copy()
                df_cluster['Cluster_Frequency'] = kmeans_frequency.fit_predict(X_cluster)
                
                clustered_dfs.append(df_cluster)
            
               # Save Frequency clustering results to CSV
            clustered_df = pd.concat(clustered_dfs, ignore_index=True)
            output_clusters_csv_path = 'kmeans_frequency_clusters.csv'  # Adjust the file path as needed
            clustered_df.to_csv(output_clusters_csv_path, index=False)
            
            # Plotting - Frequency
            for idx, df_cluster in enumerate(clustered_dfs):
                plt.figure(figsize=(10, 8))
                plt.scatter(df_cluster['Frequency (Hz)'], df_cluster['Angle of Arrival (degrees)'],
                            c=df_cluster['Cluster_Frequency'], cmap='viridis', s=50)
                plt.title(f'K-Means Clustering (Frequency) - Cluster {idx+1}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Angle of Arrival (degrees)')
                plt.colorbar(label='Cluster')
                plt.grid(True)
                plt.show()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during K-Means clustering: {str(e)}")

    def dbscan_clustering(self, file_path, eps_angle, min_samples_angle, eps_frequency, min_samples_frequency):
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
            
            # Perform DBSCAN clustering on Angle of Arrival
            dbscan_angle = DBSCAN(eps=eps_angle, min_samples=min_samples_angle)
            df['Cluster_Angle'] = dbscan_angle.fit_predict(X_scaled)

            # Save DBSCAN results to CSV
            df.to_csv('dbscan_output.csv', index=False)  # Adjust the file path as needed
        
            
            # Plotting - DBSCAN Clustering on Angle of Arrival
            plt.figure(figsize=(10, 8))
            plt.scatter(df['Angle of Arrival (degrees)'], df['Frequency (Hz)'], c=df['Cluster_Angle'], cmap='viridis', s=50)
            plt.title('DBSCAN Clustering (Angle of Arrival)')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Cluster')
            plt.grid(True)
            plt.show()
            
            # Initialize list to store clustered dataframes
            clustered_dfs = []
            
            # Perform DBSCAN clustering on Frequency for each cluster from Angle clustering
            for cluster_angle in np.unique(df['Cluster_Angle']):
                if cluster_angle == -1:
                    continue
                
                mask = df['Cluster_Angle'] == cluster_angle
                X_cluster = df.loc[mask, ['Frequency (Hz)']]
                
                if len(X_cluster) == 0:
                    continue
                
                dbscan_frequency = DBSCAN(eps=eps_frequency, min_samples=min_samples_frequency)
                df_cluster = df.loc[mask].copy()
                df_cluster['Cluster_Frequency'] = dbscan_frequency.fit_predict(X_cluster)
                
                clustered_dfs.append(df_cluster)
            
            # Save Frequency clustering results to CSV
            clustered_df = pd.concat(clustered_dfs, ignore_index=True)
            clustered_df.to_csv('dbscan_frequency_clusters.csv', index=False)  # Adjust the file path as needed
        
            # Plotting - DBSCAN Clustering on Frequency
            for idx, df_cluster in enumerate(clustered_dfs):
                plt.figure(figsize=(10, 8))
                plt.scatter(df_cluster['Frequency (Hz)'], df_cluster['Angle of Arrival (degrees)'],
                            c=df_cluster['Cluster_Frequency'], cmap='viridis', s=50)
                plt.title(f'DBSCAN Clustering (Frequency) - Cluster {idx+1}')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Angle of Arrival (degrees)')
                plt.colorbar(label='Cluster')
                plt.grid(True)
                plt.show()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during DBSCAN clustering: {str(e)}")

    def runCentroidClustering(self):
        try:
            if not hasattr(self, 'file_path'):
                raise ValueError("No CSV file selected.")
            
            if self.checkBox_5.isChecked():  # Centroid Clustering
                threshold_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Angle Threshold for Centroid Clustering", "Angle Threshold:", 5.0, 0.1)
                threshold_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Frequency Threshold for Centroid Clustering", "Frequency Threshold:", 100.0, 1.0)
                centroid_clustering(self.file_path, threshold_angle, threshold_frequency)
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during Centroid Clustering: {str(e)}")
def read_csv(file_path):
    df = pd.read_csv(file_path)
    if 'Frequency (Hz)' not in df.columns or 'Angle of Arrival (degrees)' not in df.columns:
        raise ValueError("CSV file must contain 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns.")
    return df

def calculate_mean(cluster):
    if not cluster:
        return None
    df_cluster = pd.DataFrame(cluster)
    mean_frequency = df_cluster['Frequency (Hz)'].mean()
    mean_angle = df_cluster['Angle of Arrival (degrees)'].mean()
    return {'Frequency (Hz)': mean_frequency, 'Angle of Arrival (degrees)': mean_angle}

def angle_based_clustering(df, threshold_angle):
    clusters = []
    current_cluster = []
    last_angle = None

    for i in range(len(df)):
        row = df.iloc[i]
        if last_angle is None or abs(row['Angle of Arrival (degrees)'] - last_angle) > threshold_angle:
            if current_cluster:
                clusters.append(pd.DataFrame(current_cluster))
            current_cluster = [row]
        else:
            mean_point = calculate_mean(current_cluster)
            if abs(row['Angle of Arrival (degrees)'] - mean_point['Angle of Arrival (degrees)']) < threshold_angle:
                current_cluster.append(row)
            else:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [row]
        last_angle = row['Angle of Arrival (degrees)']
    
    if current_cluster:
        clusters.append(pd.DataFrame(current_cluster))
    
    return clusters

def frequency_based_clustering(df, threshold_frequency):
    clusters = []
    current_cluster = []
    last_frequency = None

    for i in range(len(df)):
        row = df.iloc[i]
        if last_frequency is None or abs(row['Frequency (Hz)'] - last_frequency) >= threshold_frequency:
            if current_cluster:
                clusters.append(pd.DataFrame(current_cluster))
            current_cluster = [row]
        else:
            mean_point = calculate_mean(current_cluster)
            if abs(row['Frequency (Hz)'] - mean_point['Frequency (Hz)']) < threshold_frequency:
                current_cluster.append(row)
            else:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [row]
        last_frequency = row['Frequency (Hz)']
    
    if current_cluster:
        clusters.append(pd.DataFrame(current_cluster))
    
    return clusters

def plot_clusters(angle_clusters, frequency_clusters, threshold_angle, threshold_frequency):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Angle Clusters
    for angle_cluster_id, angle_cluster in enumerate(angle_clusters):
        ax1.scatter(angle_cluster['Frequency (Hz)'], angle_cluster['Angle of Arrival (degrees)'],
                    label=f'Angle Cluster {angle_cluster_id + 1}', alpha=0.7)
    
    ax1.set_xlabel('Frequency (Hz)', fontsize=14)
    ax1.set_ylabel('Angle of Arrival (degrees)', fontsize=14)
    ax1.set_title(f'Angle of Arrival-based Clustering (Threshold={threshold_angle})', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=12)

    # Plot Frequency Clusters within Angle Clusters
    for frequency_cluster_id, frequency_cluster in enumerate(frequency_clusters):
        ax2.scatter(frequency_cluster['Frequency (Hz)'], frequency_cluster['Angle of Arrival (degrees)'],
                    label=f'Freq Cluster {frequency_cluster_id + 1}', alpha=0.7)
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=14)
    ax2.set_ylabel('Angle of Arrival (degrees)', fontsize=14)
    ax2.set_title(f'Frequency-based Clustering (Threshold={threshold_frequency})', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=12)

    plt.tight_layout()

    # Save plot as PNG
    output_dir = "centroid_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_output_file_path = os.path.join(output_dir, f"cluster_plot_angle_{threshold_angle}_frequency_{threshold_frequency}.png")
    plt.savefig(plot_output_file_path)
    print(f"Plot saved as {plot_output_file_path}")

    plt.show()

def centroid_clustering(file_path, threshold_angle, threshold_frequency):
    df = read_csv(file_path)
    
    # Step 1: Cluster based on angle of arrival
    angle_clusters = angle_based_clustering(df, threshold_angle)
    
    # Step 2: For each angle cluster, cluster based on frequency
    all_frequency_clusters = []
    for angle_cluster_df in angle_clusters:
        frequency_clusters = frequency_based_clustering(angle_cluster_df, threshold_frequency)
        all_frequency_clusters.extend(frequency_clusters)
    
    plot_clusters(angle_clusters, all_frequency_clusters, threshold_angle, threshold_frequency)

      # Convert clusters to DataFrame for saving to CSV
    angle_clustered_df = pd.concat(angle_clusters, ignore_index=True)
    frequency_clustered_df = pd.concat(all_frequency_clusters, ignore_index=True)
        
    # Save centroid clustering results to CSV
    angle_clustered_df.to_csv('centroid_angle_clusters.csv', index=False)  # Adjust the file path as needed
    frequency_clustered_df.to_csv('centroid_frequency_clusters.csv', index=False)  # Adjust the file path as needed
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ClusteringApp()
    MainWindow.show()
    sys.exit(app.exec_())
