
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# App configuration
st.set_page_config(page_title="🧠 Smart Customer Segmentation", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
    <style>
        .main {background-color: #f7f9fb;}
        .block-container {padding-top: 2rem;}
        h1, h2, h3 {color: #003366;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🛍️ Smart Customer Segmentation Web App")

# Sidebar Navigation
with st.sidebar:
    st.title("🧭 Navigation")
    section = st.radio("Go to:", ["Upload Data", "Elbow Curve", "Cluster Visualization", "Explore Clusters", "Chat Assistant"])

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Upload Data Section
if section == "Upload Data":
    uploaded_file = st.file_uploader("Upload your dataset (e.g. Mall_Customers.csv)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # Preprocess
        le = LabelEncoder()
        if 'Gender' in df.columns:
            df['Gender'] = le.fit_transform(df['Gender'])

        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.session_state.X_scaled = X_scaled
        st.success("✅ Dataset loaded and processed!")

        with st.expander("🔍 View Raw Data"):
            st.dataframe(df.head())

# Use stored data
df = st.session_state.df
X_scaled = st.session_state.X_scaled

# Elbow Curve
if section == "Elbow Curve":
    st.markdown("## 📉 Elbow Method to Determine Optimal Clusters")
    if df is not None and X_scaled is not None:
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title("Elbow Curve")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload and process data in 'Upload Data' section first.")

# Cluster Visualization
if section == "Cluster Visualization":
    st.markdown("## 📊 Cluster Visualization with PCA")
    if df is not None and X_scaled is not None:
        k = st.slider("Select number of clusters", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        df['PCA1'], df['PCA2'] = pcs[:, 0], pcs[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🧑‍🤝‍🧑 Total Customers", len(df))
        with col2:
            st.metric("🔢 Total Clusters", df['Cluster'].nunique())

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax)
        ax.set_title("Customer Segments in 2D Space", fontsize=14)
        st.pyplot(fig)

        st.download_button("⬇️ Download Segmented Data", data=df.to_csv(index=False).encode('utf-8'),
                           file_name="Clustered_Customers.csv", mime="text/csv")
    else:
        st.warning("⚠️ Please upload and process data in 'Upload Data' section first.")

# Explore Clusters
if section == "Explore Clusters":
    st.markdown("## 🔍 Explore Clusters")
    if df is not None and 'Cluster' in df.columns:
        cluster_id = st.selectbox("Choose a cluster", sorted(df['Cluster'].unique()))
        cluster_df = df[df['Cluster'] == cluster_id]
        st.write(f"Showing {len(cluster_df)} customers in Cluster {cluster_id}")
        st.dataframe(cluster_df.head(10))
        st.write("📊 Average Feature Values:")
        st.dataframe(cluster_df[features].mean().to_frame("Average").T)
    else:
        st.warning("⚠️ Please generate clusters in 'Cluster Visualization' first.")

# Chat Assistant
if section == "Chat Assistant":
    st.markdown("## 💬 Ask Questions About the Segments")
    if df is not None and 'Cluster' in df.columns:
        user_input = st.text_input("Ask something like: 'Average income in cluster 2'")
        if user_input:
            response = "Sorry, I couldn't understand your question."
            tokens = user_input.lower().split()
            for i in range(len(tokens)):
                if tokens[i] in ["cluster", "group"]:
                    try:
                        cluster_num = int(tokens[i + 1])
                        if "income" in user_input:
                            avg_income = df[df['Cluster'] == cluster_num]['Annual Income (k$)'].mean()
                            response = f"Average income in cluster {cluster_num} is: {avg_income:.2f} k$"
                        elif "spending" in user_input:
                            avg_spend = df[df['Cluster'] == cluster_num]['Spending Score (1-100)'].mean()
                            response = f"Average spending score in cluster {cluster_num} is: {avg_spend:.2f}"
                        elif "age" in user_input:
                            avg_age = df[df['Cluster'] == cluster_num]['Age'].mean()
                            response = f"Average age in cluster {cluster_num} is: {avg_age:.2f}"
                        elif "how many" in user_input or "count" in user_input:
                            count = len(df[df['Cluster'] == cluster_num])
                            response = f"There are {count} customers in cluster {cluster_num}"
                    except:
                        pass
            st.session_state.chat_history.append((user_input, response))
            st.write(f"**🧠 Assistant:** {response}")

        if st.session_state.chat_history:
            with st.expander("🕘 Chat History"):
                for q, a in st.session_state.chat_history:
                    st.markdown(f"**You:** {q}")
                    st.markdown(f"**Assistant:** {a}")
    else:
        st.warning("⚠️ Please generate clusters first to enable chat.")
