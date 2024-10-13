import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D


# Existing UI Class and Functions
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 50, 191, 41))
        self.pushButton.setObjectName("pushButton")
        
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(300, 50, 191, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(50, 130, 261, 31))
        self.checkBox_2.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_2.setObjectName("checkBox_2")
        
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(300, 130, 261, 31))
        self.checkBox_3.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_3.setObjectName("checkBox_3")
        
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(50, 210, 261, 31))
        self.checkBox_4.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_4.setObjectName("checkBox_4")

        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(300, 210, 261, 31))
        self.checkBox_5.setStyleSheet("font: 87 12pt \"Outfit Black\";")
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_5.setText("Centroid Clustering")

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
        self.checkBox_4.setText(_translate("MainWindow", "Hierarchical Clustering"))
        self.checkBox_5.setText(_translate("MainWindow", "Centroid Clustering"))

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
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CSV Files (*.csv)", options=options)
        if file_name:
            self.file_path = file_name
            # Assuming you have a lineEdit widget to display the file path
            # self.lineEdit.setText(file_name)

    def generateOutput(self):
        try:
            if not hasattr(self, 'file_path'):
                raise ValueError("No CSV file selected.")
            
            if self.checkBox_2.isChecked():  # K-Means
                n_clusters, _ = QtWidgets.QInputDialog.getInt(None, "Enter Number of Clusters for K-Means", "Number of Clusters:", 3, 1)
                self.kmeans_clustering(self.file_path, n_clusters)
            
            if self.checkBox_3.isChecked():  # DBSCAN
                eps, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Eps for DBSCAN", "Eps:", 0.5, 0.01)
                min_samples, _ = QtWidgets.QInputDialog.getInt(None, "Enter Min Samples for DBSCAN", "Min Samples:", 5, 1)
                self.dbscan_clustering(self.file_path, eps, min_samples)
            
            if self.checkBox_4.isChecked():  # Hierarchical Clustering
                n_clusters, _ = QtWidgets.QInputDialog.getInt(None, "Enter Number of Clusters for Hierarchical Clustering", "Number of Clusters:", 3, 1)
                self.hierarchical_clustering(self.file_path, n_clusters)

            if self.checkBox_5.isChecked():  # Centroid Clustering
                threshold_angle, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Angle Threshold for Centroid Clustering", "Angle Threshold:", 5.0, 0.1)
                threshold_frequency, _ = QtWidgets.QInputDialog.getDouble(None, "Enter Frequency Threshold for Centroid Clustering", "Frequency Threshold:", 100.0, 1.0)
                centroid_clustering(self.file_path, threshold_angle, threshold_frequency)
        
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

    def hierarchical_clustering(self, file_path, n_clusters):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Select columns for clustering
            cols_selected = ['Frequency (Hz)', 'Angle of Arrival (degrees)']
            
            # Extract data for clustering
            data_to_cluster = df[cols_selected].values
            
            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_to_cluster)
            
            # Perform hierarchical clustering
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            clusters = model.fit_predict(data_scaled)
            
            # Calculate median values for each cluster
            cluster_medians = pd.DataFrame(data_to_cluster, columns=cols_selected)
            cluster_medians['Cluster'] = clusters
            cluster_medians = cluster_medians.groupby('Cluster').median()
            
            # Save cluster medians to CSV
            medians_filename = 'cluster_medians.csv'
            cluster_medians.to_csv(medians_filename, index=False)
            print(f"Median values of clusters saved to {medians_filename}")
            
            # Add cluster labels to original data and save to CSV
            output_filename = 'output_hierarchical_clusters.csv'
            output_data = pd.DataFrame({
                'Frequency (Hz)': df['Frequency (Hz)'],
                'Angle of Arrival (degrees)': df['Angle of Arrival (degrees)'],
                'Cluster': clusters
            })
            output_data.to_csv(output_filename, index=False)
            print(f"Hierarchical cluster labels saved to {output_filename}")
            
            # Compute silhouette scores
            silhouette_avg = silhouette_score(data_scaled, clusters)
            sample_silhouette_values = silhouette_samples(data_scaled, clusters)
            
            # Print average silhouette score
            print(f"Average silhouette score: {silhouette_avg:.2f}")
            
            # Plotting (you can add plotting function here if needed)
            self.plot_clusters(df, cols_selected, clusters, silhouette_avg, sample_silhouette_values)
            
            # Plot dendrogram and save linkage matrix
            linkage_matrix = linkage(data_scaled, method='ward')
            plt.figure(figsize=(12, 8))
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Sample Index')
            plt.ylabel('Distance')
            dendrogram(linkage_matrix)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during hierarchical clustering: {str(e)}")

    def plot_clusters(self, data, cols_selected, clusters, silhouette_avg, sample_silhouette_values):
        try:
            fig = plt.figure(figsize=(15, 7))
            
            # 3D Scatter plot
            ax1 = fig.add_subplot(121, projection='3d')
            for cluster in np.unique(clusters):
                cluster_data = data[clusters == cluster]
                ax1.scatter(cluster_data[cols_selected[0]], 
                            cluster_data[cols_selected[1]], 
                            zs=cluster,
                            label=f'Cluster {cluster}',
                            s=50,
                            alpha=0.8)
            
            ax1.set_xlabel(cols_selected[0])
            ax1.set_ylabel(cols_selected[1])
            ax1.set_zlabel('Cluster')
            ax1.set_title('Hierarchical Clustering - 3D Scatter Plot')
            ax1.legend()
            
            # Silhouette plot
            ax2 = fig.add_subplot(122)
            y_lower = 10
            for i, cluster in enumerate(np.unique(clusters)):
                cluster_silhouette_values = sample_silhouette_values[clusters == cluster]
                cluster_silhouette_values.sort()
                
                size_cluster_i = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                ax2.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, cluster_silhouette_values,
                                  alpha=0.7, label=f'Cluster {cluster}')
                
                ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
                
                y_lower = y_upper + 10
            
            ax2.set_title(f"Silhouette plot for each cluster (Avg Silhouette Score: {silhouette_avg:.2f})")
            ax2.set_xlabel("Silhouette coefficient values")
            ax2.set_ylabel("Cluster label")
            
            # The vertical line for average silhouette score of all the values
            ax2.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax2.set_yticks([])
            ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            
            plt.suptitle("Hierarchical Clustering with Silhouette Analysis")
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error occurred during plotting: {str(e)}")

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

# New functions for Centroid Clustering
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
        if last_angle is None or abs(row['Angle of Arrival (degrees)'] - last_angle) >= threshold_angle:
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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ClusteringApp()
    window.show()
    sys.exit(app.exec_())