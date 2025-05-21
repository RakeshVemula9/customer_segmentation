import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Smart Customer Segmentation App")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("Upload your CSV (e.g. Mall_Customers.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.dataframe(df.head())

    # Preprocess
    le = LabelEncoder()
    if 'Gender' in df.columns:
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

    with st.expander("📉 Elbow Method"):
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title("Elbow Curve")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

    # Cluster input
    k = st.slider("Select number of clusters", 2, 10, 5)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for cluster plotting
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = pcs[:, 0], pcs[:, 1]

    st.subheader("📊 Visualize Clusters")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax2)
    st.pyplot(fig2)

    # Cluster Explorer
    st.subheader("🔎 Explore a Cluster")
    cluster_id = st.selectbox("Select a cluster to explore", sorted(df['Cluster'].unique()))
    cluster_df = df[df['Cluster'] == cluster_id]
    st.write(f"Showing {len(cluster_df)} customers in Cluster {cluster_id}")
    st.dataframe(cluster_df.head(10))
    st.write("Average values:")
    st.dataframe(cluster_df[features].mean().to_frame("Average").T)

    # Download results
    st.download_button(
        label="⬇️ Download Segmented Data",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="Clustered_Customers.csv",
        mime="text/csv"
    )

    # Simple rule-based chatbot assistant
    st.subheader("💬 Ask About Clusters")
    user_input = st.text_input("Ask a question about the data (e.g., 'Average income in cluster 2')")

    if user_input:
        response = "Sorry, I couldn't understand your question."
        tokens = user_input.lower().split()
        for i in range(len(tokens)):
            if tokens[i] in ["cluster", "group"]:
                try:
                    cluster_num = int(tokens[i+1])
                    if "income" in user_input:
                        response = f"Average income in cluster {cluster_num} is: {df[df['Cluster']==cluster_num]['Annual Income (k$)'].mean():.2f} k$"
                    elif "spending" in user_input:
                        response = f"Average spending score in cluster {cluster_num} is: {df[df['Cluster']==cluster_num]['Spending Score (1-100)'].mean():.2f}"
                    elif "age" in user_input:
                        response = f"Average age in cluster {cluster_num} is: {df[df['Cluster']==cluster_num]['Age'].mean():.2f}"
                    elif "how many" in user_input or "count" in user_input:
                        response = f"There are {len(df[df['Cluster']==cluster_num])} customers in cluster {cluster_num}"
                except:
                    pass

        st.session_state.chat_history.append((user_input, response))
        st.write(f"**🧠 Assistant:** {response}")

    # Display chat history
    if st.session_state.chat_history:
        with st.expander("🕘 Chat History"):
            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Assistant:** {a}")

