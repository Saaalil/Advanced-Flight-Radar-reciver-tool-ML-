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
        self.pushButton.setText(_translate("MainWindow", "Input File"))
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
        self.confirm_dialog = CustomDialog(self)  # Initialize the confirmation dialog

    def selectFile(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv);;JSON Files (*.json);;Excel Files (*.xlsx)", options=options)
        if file_name:
            self.file_path = file_name
            QtWidgets.QMessageBox.information(self, "File Selected", f"Selected file: {self.file_path}")
      



    def generateOutput(self):
        try:
            if not self.file_path:
                raise ValueError("No file selected.")
            
            if self.checkBox_2.isChecked():  # K-Means
                n_clusters_angle, _ = QtWidgets.QInputDialog.getInt(None, "Enter Number of Clusters for K-Means (Angle)", "Number of Clusters (Angle):", 3, 1)
                n_clusters_frequency, _ = QtWidgets.QInputDialog.getInt(None, "Enter Number of Clusters for K-Means (Frequency)", "Number of Clusters (Frequency):", 3, 1)
                save_path, filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save Output", "", "JSON Files (*.json);;Excel Files (*.xlsx);;CSV Files (*.csv)")
                if save_path:
                    if filter == "JSON Files (*.json)":
                        save_path = self.check_extension(save_path, ".json")
                    elif filter == "Excel Files (*.xlsx)":
                        save_path = self.check_extension(save_path, ".xlsx")
                    elif filter == "CSV Files (*.csv)":
                        save_path = self.check_extension(save_path, ".csv")
                    self.kmeans_clustering(self.file_path, n_clusters_angle, n_clusters_frequency, save_path)
                    QtWidgets.QMessageBox.information(None, "Data Generated", "Clustering data has been successfully generated.")
                    QtWidgets.QMessageBox.information(None, "Data Saved", "Clustering results have been successfully saved.")

            if self.checkBox_3.isChecked():  # DBSCAN
                eps_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN (Angle)", "Eps (Angle):", 0.5, 0.01)
                min_samples_angle, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN (Angle)", "Min Samples (Angle):", 5, 1)
                eps_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN (Frequency)", "Eps (Frequency):", 0.5, 0.01)
                min_samples_frequency, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN (Frequency)", "Min Samples (Frequency):", 5, 1)
                
                save_path, filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save Output", "", "JSON Files (*.json);;Excel Files (*.xlsx);;CSV Files (*.csv)")
                if save_path:
                    if filter == "JSON Files (*.json)":
                        save_path = self.check_extension(save_path, ".json")
                    elif filter == "Excel Files (*.xlsx)":
                        save_path = self.check_extension(save_path, ".xlsx")
                    elif filter == "CSV Files (*.csv)":
                        save_path = self.check_extension(save_path, ".csv")
                    
                    self.dbscan_clustering(self.file_path, eps_angle, min_samples_angle, eps_frequency, min_samples_frequency, save_path)
                    QtWidgets.QMessageBox.information(None, "Data Generated", "Clustering data has been successfully generated.")
                    QtWidgets.QMessageBox.information(None, "Data Saved", "Clustering results have been successfully saved.")
            if self.checkBox_5.isChecked():  # Centroid Clustering
                threshold_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Angle Threshold for Centroid Clustering", "Angle Threshold:", 5.0, 0.1)
                threshold_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Frequency Threshold for Centroid Clustering", "Frequency Threshold:", 100.0, 1.0)
                centroid_clustering(self.file_path, threshold_angle, threshold_frequency)

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred: {str(e)}")
            
            
    def closeEvent(self, event):
        reply = self.confirm_dialog.exec_()
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

            
            
    def read_csv_and_find_freq_unit(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please select a CSV, JSON, or Excel file.")
        
        # Find the column that contains 'Frequency'
        freq_column = [col for col in df.columns if 'Frequency' in col]
        
        if len(freq_column) != 1:
            raise ValueError("Could not determine the frequency unit column.")
        
        freq_unit = freq_column[0].split('(')[-1].split(')')[0]  # Extracting the unit from parentheses
        
        
        return df, freq_unit
        

    
    
    
    
    
    def check_extension(self, file_path, extension):
        if not file_path.endswith(extension):
            file_path += extension
        return file_path

    

    def kmeans_clustering(self, file_path, n_clusters_angle, n_clusters_frequency, save_path):
        try:
            # Read CSV file 
            df, freq_unit = self.read_csv_and_find_freq_unit(file_path)

            # Perform K-Means clustering on Angle of Arrival
            kmeans_angle = KMeans(n_clusters=n_clusters_angle, random_state=42)
            df['Cluster_Angle'] = kmeans_angle.fit_predict(df[['Angle of Arrival (degrees)']])

            # Get centroids for Angle of Arrival clusters
            centroids_angle = kmeans_angle.cluster_centers_

            # Extract frequency column dynamically based on the found frequency unit
            freq_column = f'Frequency ({freq_unit})'

            # Perform K-Means clustering on Frequency for each cluster from Angle clustering
            clustered_dfs = []
            centroids_frequency_dict = {}
            for cluster_angle in range(n_clusters_angle):
                mask = df['Cluster_Angle'] == cluster_angle
                X_cluster = df.loc[mask, [freq_column]]

                if len(X_cluster) == 0:
                    continue

                kmeans_frequency = KMeans(n_clusters=n_clusters_frequency, random_state=42)
                df_cluster = df.loc[mask].copy()
                df_cluster['Cluster_Frequency'] = kmeans_frequency.fit_predict(X_cluster)

                # Get centroids for Frequency clusters within each Angle cluster
                centroids_frequency = kmeans_frequency.cluster_centers_
                centroids_frequency_dict[cluster_angle] = centroids_frequency

                # Assign centroids to df_cluster
                df_cluster['Centroid_Angle'] = centroids_angle[cluster_angle, 0]  # Angle centroid
                for freq_cluster in range(n_clusters_frequency):
                    freq_mask = df_cluster['Cluster_Frequency'] == freq_cluster
                    df_cluster.loc[freq_mask, 'Centroid_Frequency'] = centroids_frequency[freq_cluster, 0]  # Frequency centroid

                clustered_dfs.append(df_cluster)

            # Concatenate all clustered dataframes
            clustered_df = pd.concat(clustered_dfs)

            # Prepare centroids DataFrame
            centroids_data = {
                'Angle of Arrival (degrees)': [],
                freq_column: [],
                'Cluster_Angle': [],
                'Cluster_Frequency': [],
                'Centroid_Angle': [],
                'Centroid_Frequency': []
            }

            for cluster_angle, freq_centroids in centroids_frequency_dict.items():
                for i, freq_centroid in enumerate(freq_centroids):
                    centroids_data['Angle of Arrival (degrees)'].append(None)
                    centroids_data[freq_column].append(None)
                    centroids_data['Cluster_Angle'].append(cluster_angle)
                    centroids_data['Cluster_Frequency'].append(i)
                    centroids_data['Centroid_Angle'].append(centroids_angle[cluster_angle, 0])
                    centroids_data['Centroid_Frequency'].append(freq_centroid[0])

            centroids_df = pd.DataFrame(centroids_data)

            # Drop duplicates for centroid information
            centroids_df = centroids_df.drop_duplicates(subset=['Cluster_Angle', 'Cluster_Frequency'])

            # Combine clustered_df and centroids_df
            result_df = pd.concat([clustered_df, centroids_df], ignore_index=True)

            # Save results to CSV
            result_df.to_csv(save_path, index=False)

            # Plotting - Angle of Arrival
            plt.figure(figsize=(10, 8))
            plt.scatter(df['Angle of Arrival (degrees)'], df[freq_column], c=df['Cluster_Angle'], cmap='viridis', s=50)
            plt.title('K-Means Clustering (Angle of Arrival)')
            plt.xlabel('Angle of Arrival (degrees)')
            plt.ylabel(freq_column)
            plt.colorbar(label='Cluster')
            plt.grid(True)
            plt.savefig('kmeans_angle_plot.png')
            plt.show()

            # Plotting - Frequency
            for idx, df_cluster in enumerate(clustered_dfs):
                plt.figure(figsize=(10, 8))
                plt.scatter(df_cluster[freq_column], df_cluster['Angle of Arrival (degrees)'],
                            c=df_cluster['Cluster_Frequency'], cmap='viridis', s=50)
                plt.title(f'K-Means Clustering (Frequency) - Cluster {idx + 1}')
                plt.xlabel(freq_column)
                plt.ylabel('Angle of Arrival (degrees)')
                plt.colorbar(label='Cluster')
                plt.grid(True)
                plt.savefig(f'kmeans_frequency_plot_cluster_{idx + 1}.png')
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
            df.to_csv(save_path, index=False)  # Adjust the file path as needed
        
            
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
    """Calculate the mean of the 'Frequency (Hz)' and 'Angle of Arrival (degrees)' columns for a cluster."""
    if cluster.empty:
        return None
    mean_frequency = cluster['Frequency (Hz)'].mean()
    mean_angle = cluster['Angle of Arrival (degrees)'].mean()
    return {'Frequency (Hz)': mean_frequency, 'Angle of Arrival (degrees)': mean_angle}

def angle_based_clustering(df, threshold_angle):
    """Cluster data points based on the angle of arrival."""
    clusters = []

    for i in range(len(df)):
        row = df.iloc[[i]]
        added_to_cluster = False

        # Check against all existing clusters
        for j in range(len(clusters)):
            cluster = clusters[j]
            mean_point = calculate_mean(cluster)
            if abs(row['Angle of Arrival (degrees)'].values[0] - mean_point['Angle of Arrival (degrees)']) < threshold_angle:
                clusters[j] = pd.concat([cluster, row])
                added_to_cluster = True
                break

        if not added_to_cluster:
            clusters.append(row)
    
    return clusters

def frequency_based_clustering(df, threshold_frequency):
    """Cluster data points based on frequency."""
    clusters = []

    for i in range(len(df)):
        row = df.iloc[[i]]
        added_to_cluster = False

        # Check against all existing clusters
        for j in range(len(clusters)):
            cluster = clusters[j]
            mean_point = calculate_mean(cluster)
            if abs(row['Frequency (Hz)'].values[0] - mean_point['Frequency (Hz)']) < threshold_frequency:
                clusters[j] = pd.concat([cluster, row])
                added_to_cluster = True
                break

        if not added_to_cluster:
            clusters.append(row)
    
    return clusters

def plot_clusters(angle_clusters, frequency_clusters, threshold_angle, threshold_frequency):
    """Plot the angle-based and frequency-based clusters."""
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

def save_clusters_to_files(clustered_df, threshold_angle, threshold_frequency):
    """Save clustered data to CSV, JSON, and XLSX files."""
    output_dir = "centroid_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save as CSV
    csv_file_path = os.path.join(output_dir, f"clustered_data_angle_{threshold_angle}_frequency_{threshold_frequency}.csv")
    clustered_df.to_csv(csv_file_path, index=False)
    print(f"Clustered data saved to {csv_file_path}")

    # Save as JSON
    json_file_path = os.path.join(output_dir, f"clustered_data_angle_{threshold_angle}_frequency_{threshold_frequency}.json")
    clustered_df.to_json(json_file_path, orient='records', lines=True)
    print(f"Clustered data saved to {json_file_path}")

    # Save as Excel
    excel_file_path = os.path.join(output_dir, f"clustered_data_angle_{threshold_angle}_frequency_{threshold_frequency}.xlsx")
    clustered_df.to_excel(excel_file_path, index=False)
    print(f"Clustered data saved to {excel_file_path}")

def centroid_clustering(file_path, threshold_angle, threshold_frequency):
    """Perform centroid clustering based on angle and frequency thresholds."""
    df = read_csv(file_path)
    
    # Step 1: Cluster based on angle of arrival
    angle_clusters = angle_based_clustering(df, threshold_angle)
    
    # Step 2: For each angle cluster, cluster based on frequency
    all_frequency_clusters = []
    for angle_cluster_df in angle_clusters:
        frequency_clusters = frequency_based_clustering(angle_cluster_df, threshold_frequency)
        all_frequency_clusters.extend(frequency_clusters)
    
    # Prepare data for saving
    clustered_data = []
    for freq_cluster in all_frequency_clusters:
        mean_point = calculate_mean(freq_cluster)
        clustered_data.append(mean_point)
    
    clustered_df = pd.DataFrame(clustered_data)
    
    # Plot the clusters
    plot_clusters(angle_clusters, all_frequency_clusters, threshold_angle, threshold_frequency)
    
    # Save the clustered data to files
    save_clusters_to_files(clustered_df, threshold_angle, threshold_frequency)
        
    # Save centroid clustering results to CSV
class CustomDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)        
    # Save centroid clustering results to CSV
class CustomDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    def exec_(self):
        reply = QtWidgets.QMessageBox.question(self, 'Message',
            "Are you sure you want to close this window?", QtWidgets.QMessageBox.Yes |
            QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return reply

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ClusteringApp()
    MainWindow.show()
    sys.exit(app.exec_())
