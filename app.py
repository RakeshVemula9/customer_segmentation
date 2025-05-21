import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("🛍️ Customer Segmentation using Machine Learning")

# Upload dataset
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    # Preprocessing
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    st.subheader("📉 Elbow Method for Optimal Clusters")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    # Choose clusters
    n_clusters = st.slider("Choose number of clusters", 2, 10, 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df['PCA1'] = pcs[:, 0]
    df['PCA2'] = pcs[:, 1]

    st.subheader("📊 Cluster Visualization")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2', ax=ax2)
    st.pyplot(fig2)

    st.subheader("📋 Cluster Summary")
    st.dataframe(df.groupby('Cluster')[features].mean())

    st.download_button(
        label="Download Clustered Data as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='Segmented_Customers.csv',
        mime='text/csv'
    )
