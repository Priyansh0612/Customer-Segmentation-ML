import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

# Load the cleaned dataset (from Step 1)
file_path = "../data/processed/Updated_Project1.csv"
df = pd.read_csv(file_path)

# ----------------------------------------------
# Step 3: Segment Customers Using Clustering
# ----------------------------------------------
def segment_customers(df):
    """Apply preprocessing, cluster customers using two techniques, and visualize/explain the segments."""
    
    print("\nStep 3: Segmenting Customers Using Clustering")

    # 3.1 Preprocessing for Clustering
    # Convert 'Frequency of Purchases' to numeric if categorical
    if 'Frequency of Purchases' in df.columns and df['Frequency of Purchases'].dtype == 'object':
        frequency_mapping = {"Daily": 7, "Weekly": 1, "Fortnightly": 0.5, "Monthly": 0.25}
        df['Frequency of Purchases'] = df['Frequency of Purchases'].map(frequency_mapping)
        print("- Converted 'Frequency of Purchases' to numeric values.")

    # Select features for clustering
    clustering_features = ['Age', 'Purchase Amount (USD)', 'Frequency of Purchases']
    df_cluster = df[clustering_features].dropna()
    print(f"- Selected features for clustering: {clustering_features}")
    print(f"- Rows available for clustering after dropping NaN: {len(df_cluster)}")

    # Standardize the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    print("- Features standardized using StandardScaler.")

    # 3.2 Apply Clustering Techniques
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)
    print("- K-Means clustering applied with 3 clusters.")

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=3)
    df_cluster['Agglo_Cluster'] = agglo.fit_predict(df_scaled)
    print("- Agglomerative clustering applied with 3 clusters.")

    # 3.3 Visualize Customer Segments
    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    df_cluster_pca = pca.fit_transform(df_scaled)
    df_cluster['PCA1'] = df_cluster_pca[:, 0]
    df_cluster['PCA2'] = df_cluster_pca[:, 1]
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"- PCA applied: {explained_variance:.2%} of variance explained by 2 components.")

    # Visualize K-Means Clusters
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df_cluster['PCA1'], y=df_cluster['PCA2'], hue=df_cluster['KMeans_Cluster'], palette='viridis')
    plt.title("Customer Segments Using K-Means Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    # Visualize Agglomerative Clusters
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df_cluster['PCA1'], y=df_cluster['PCA2'], hue=df_cluster['Agglo_Cluster'], palette='coolwarm')
    plt.title("Customer Segments Using Agglomerative Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    # 3.4 Explain Cluster Characteristics
    # K-Means Cluster Characteristics
    print("\nK-Means Cluster Characteristics:")
    for cluster in df_cluster['KMeans_Cluster'].unique():
        segment = df_cluster[df_cluster['KMeans_Cluster'] == cluster]
        print(f"\nCluster {cluster} (Size: {len(segment)} customers):")
        print(f"- Average Age: {segment['Age'].mean():.2f}")
        print(f"- Average Purchase Amount: ${segment['Purchase Amount (USD)'].mean():.2f}")
        print(f"- Average Purchase Frequency: {segment['Frequency of Purchases'].mean():.2f}")

    # Agglomerative Cluster Characteristics
    print("\nAgglomerative Cluster Characteristics:")
    for cluster in df_cluster['Agglo_Cluster'].unique():
        segment = df_cluster[df_cluster['Agglo_Cluster'] == cluster]
        print(f"\nCluster {cluster} (Size: {len(segment)} customers):")
        print(f"- Average Age: {segment['Age'].mean():.2f}")
        print(f"- Average Purchase Amount: ${segment['Purchase Amount (USD)'].mean():.2f}")
        print(f"- Average Purchase Frequency: {segment['Frequency of Purchases'].mean():.2f}")

    return df_cluster

# Execute Step 3
if __name__ == "__main__":
    df_cluster = segment_customers(df)
    print("\nCustomer segmentation completed.")