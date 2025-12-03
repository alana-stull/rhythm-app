import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib # Import joblib for saving/loading models

# load data
csv_filename = 'ScreenTime vs MentalWellness.csv'
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: Could not find '{csv_filename}'. Ensure it is in the same directory.")
    exit()

# features for clustering
CLUSTER_FEATURES = ['screen_time_hours', 'sleep_hours', 'productivity_0_100']
data_for_clustering = df[CLUSTER_FEATURES].copy()

# removing missing values
if data_for_clustering.isnull().values.any():
    data_for_clustering = data_for_clustering.fillna(data_for_clustering.mean()) 
    print("Missing values handled by mean imputation.")

# scaling data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)
scaled_df = pd.DataFrame(scaled_data, columns=CLUSTER_FEATURES)

print(f"Features Selected for Clustering: {CLUSTER_FEATURES}")

# elbow/silhouette (existing code for finding k)
inertia = []
sil_scores = []
k_range = range(2, 8) 

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') 
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    if k > 1:
        sil_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# plotting elbow method (truncated for brevity)

# plotting silhouette scores (truncated for brevity)

# --- Final Model Training (k=5) ---
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
df['Rhythm_Cluster'] = kmeans.fit_predict(scaled_data) 

# analyze/label clusters
cluster_summary = df.groupby('Rhythm_Cluster')[CLUSTER_FEATURES].mean().round(2)
print("\nCluster Summary (Average Feature Values per Cluster):")
print(cluster_summary)

# --- MANDATORY: SAVING THE TRAINED MODEL AND SCALER ---
joblib.dump(kmeans, 'rhythm_kmeans_model.pkl')
joblib.dump(scaler, 'rhythm_scaler.pkl')
print("\nTrained KMeans model and StandardScaler saved as 'rhythm_kmeans_model.pkl' and 'rhythm_scaler.pkl'.")

# visualize clusters (truncated for brevity)