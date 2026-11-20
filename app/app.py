
import streamlit as st
import joblib
import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fake News Classification & Clustering",
    page_icon="📰",
    layout="wide"
)


# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLUSTER_RESULTS_PATH = os.path.join(
    DATASET_DIR,
    "news_clustering_results.csv"
)


# ==========================================
# 3. LOAD SAVED MODELS
# ==========================================

@st.cache_resource
def load_models():

    svm_model = joblib.load(
        os.path.join(MODEL_DIR, "fake_news_svm.pkl")
    )

    classification_tfidf = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    kmeans_model = joblib.load(
        os.path.join(MODEL_DIR, "news_clustering_model.pkl")
    )

    clustering_tfidf = joblib.load(
        os.path.join(MODEL_DIR, "clustering_tfidf.pkl")
    )

    svd_model = joblib.load(
        os.path.join(MODEL_DIR, "clustering_svd.pkl")
    )

    return (
        svm_model,
        classification_tfidf,
        kmeans_model,
        clustering_tfidf,
        svd_model
    )


try:
    (
        svm_model,
        classification_tfidf,
        kmeans_model,
        clustering_tfidf,
        svd_model
    ) = load_models()

except Exception as e:
    st.error(f"Unable to load ML models: {e}")
    st.stop()


# ==========================================
# 4. TEXT PREPROCESSING
# ==========================================

try:
    stop_words = set(stopwords.words("english"))

except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    except Exception:
        st.error(
            "NLTK stopwords are unavailable. "
            "Run: python -m nltk.downloader stopwords"
        )
        st.stop()


stemmer = PorterStemmer()


def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word) for word in words
    ]

    return " ".join(words)


# ==========================================
# 5. LOAD CLUSTERING RESULTS
# ==========================================

@st.cache_data
def load_cluster_results():

    return pd.read_csv(CLUSTER_RESULTS_PATH)


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("📰 News Analysis")

st.sidebar.write("Fake News ML Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Fake News Detection",
        "News Clustering",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Classification + Clustering")


# ==========================================
# 7. HOME PAGE
# ==========================================

if page == "Home":

    st.title("📰 Fake News Classification and Clustering")

    st.write(
        "An ML-based application that classifies news articles "
        "as Fake or Real and groups similar articles into topics."
    )

    st.divider()

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Classification Model", "LinearSVC")
    col2.metric("Clustering Algorithm", "MiniBatch KMeans")
    col3.metric("Number of Clusters", "14")

    st.divider()

    st.subheader("Application Features")

    st.markdown("""
    **1. Fake News Detection**
    - Enter a news article.
    - Predict whether it is Fake or Real.

    **2. News Clustering**
    - Group news articles by similarity.
    - Identify the cluster and its important keywords.

    **3. About Project**
    - Explore the dataset.
    - Understand the algorithms.
    - View project evaluation results.
    """)

    st.info(
        "Select a page from the sidebar to explore the application."
    )


# ==========================================
# 8. FAKE NEWS DETECTION PAGE
# ==========================================

elif page == "Fake News Detection":

    st.title("🔍 Fake News Detection")

    st.write(
        "Enter a news article to predict whether it is Fake or Real."
    )

    news_text = st.text_area(
        "Enter News Article",
        height=250,
        placeholder="Paste your news article here..."
    )

    if st.button("Classify News", type="primary"):

        if not news_text.strip():

            st.warning("Please enter a news article.")

        else:

            cleaned_text = preprocess_text(news_text)

            if not cleaned_text.strip():

                st.warning(
                    "The article contains no usable text after preprocessing."
                )

            else:

                transformed_text = classification_tfidf.transform(
                    [cleaned_text]
                )

                prediction = svm_model.predict(
                    transformed_text
                )[0]

                st.divider()
                st.subheader("Prediction Result")

                if prediction == 0:

                    st.error(
                        "🚨 This article is classified as FAKE NEWS."
                    )

                else:

                    st.success(
                        "✅ This article is classified as REAL NEWS."
                    )

                st.caption(
                    "This is a machine-learning prediction, "
                    "not independent verification of the article's truth."
                )


# ==========================================
# 9. NEWS CLUSTERING PAGE
# ==========================================

elif page == "News Clustering":

    st.title("📊 News Article Clustering")

    st.write(
        "Enter a news article to find which of the 14 clusters "
        "it most closely belongs to."
    )

    article_text = st.text_area(
        "Enter News Article for Clustering",
        height=250,
        placeholder="Paste a news article here..."
    )

    # ======================================
    # 9A. PREDICT CLUSTER
    # ======================================

    if st.button("Find News Cluster", type="primary"):

        if not article_text.strip():

            st.warning("Please enter a news article.")

        else:

            cleaned_text = preprocess_text(article_text)

            if not cleaned_text.strip():

                st.warning(
                    "The article contains no usable text after preprocessing."
                )

            else:

                article_vector = clustering_tfidf.transform(
                    [cleaned_text]
                )

                cluster_number = int(
                    kmeans_model.predict(article_vector)[0]
                )

                st.divider()

                st.subheader("Clustering Result")

                st.success(
                    f"This article belongs to Cluster {cluster_number}."
                )

                # Extract top 15 cluster keywords
                feature_names = (
                    clustering_tfidf.get_feature_names_out()
                )

                cluster_center = (
                    kmeans_model.cluster_centers_[cluster_number]
                )

                top_indices = cluster_center.argsort()[-15:][::-1]

                top_words = [
                    feature_names[index]
                    for index in top_indices
                ]

                st.subheader("Important Cluster Keywords")

                st.write(", ".join(top_words))

                st.caption(
                    "Keywords represent the important terms "
                    "associated with this cluster."
                )

    # ======================================
    # 9B. LOAD CLUSTERING RESULTS
    # ======================================

    st.divider()

    try:

        cluster_results = load_cluster_results()

        required_columns = {
            "article",
            "x",
            "y",
            "cluster"
        }

        if not required_columns.issubset(cluster_results.columns):

            st.error(
                "Clustering results CSV must contain "
                "article, x, y and cluster columns."
            )

        else:

            # ==================================
            # 9C. CLUSTER DISTRIBUTION
            # ==================================

            st.subheader("Overall Cluster Distribution")

            cluster_counts = (
                cluster_results["cluster"]
                .value_counts()
                .sort_index()
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Articles",
                f"{len(cluster_results):,}"
            )

            col2.metric(
                "Total Clusters",
                cluster_results["cluster"].nunique()
            )

            st.bar_chart(
                cluster_counts,
                x_label="Cluster Number",
                y_label="Number of Articles"
            )

            st.caption(
                "Number of news articles assigned to each cluster."
            )

            # ==================================
            # 9D. 2D CLUSTER VISUALIZATION
            # ==================================

            st.divider()

            st.subheader("2D News Cluster Visualization")

            st.write(
                "Each dot represents one news article. "
                "Different colors represent different clusters."
            )

            fig, ax = plt.subplots(figsize=(12, 7))

            scatter = ax.scatter(
                cluster_results["x"],
                cluster_results["y"],
                c=cluster_results["cluster"],
                cmap="tab20",
                s=5,
                alpha=0.6
            )

            ax.set_xlabel("SVD Component 1")
            ax.set_ylabel("SVD Component 2")

            ax.set_title(
                "News Articles Grouped into 14 Clusters"
            )

            fig.colorbar(
                scatter,
                ax=ax,
                label="Cluster Number"
            )

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "This visualization uses the saved 2D SVD coordinates "
                "generated during model development."
            )

            # ==================================
            # 9E. CLUSTER EXPLORATION
            # ==================================

            st.divider()

            st.subheader("🔎 Explore News Clusters")

            st.write(
                "Select a cluster to explore the news articles "
                "assigned to it."
            )

            available_clusters = sorted(
                cluster_results["cluster"].dropna().unique().tolist()
            )

            selected_cluster = st.selectbox(
                "Select a Cluster",
                available_clusters,
                format_func=lambda x: f"Cluster {x}"
            )

            # Filter articles belonging to selected cluster
            selected_articles = cluster_results[
                cluster_results["cluster"] == selected_cluster
            ].copy()

            # Display cluster statistics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Cluster Number",
                int(selected_cluster)
            )

            col2.metric(
                "Total Articles",
                f"{len(selected_articles):,}"
            )

            if "label" in selected_articles.columns:

                fake_count = (
                    selected_articles["label"] == 0
                ).sum()

                real_count = (
                    selected_articles["label"] == 1
                ).sum()

                col3.metric(
                    "Fake / Real Articles",
                    f"{fake_count:,} / {real_count:,}"
                )

            # ==================================
            # 9F. DISPLAY SAMPLE ARTICLES
            # ==================================

            st.subheader("Articles in This Cluster")

            number_of_articles = st.slider(
                "Number of sample articles to display",
                min_value=1,
                max_value=10,
                value=5
            )

            sample_size = min(
                number_of_articles,
                len(selected_articles)
            )

            sample_articles = selected_articles.sample(
                n=sample_size,
                random_state=42
            )

            for index, (_, row) in enumerate(
                sample_articles.iterrows(),
                start=1
            ):

                with st.expander(f"Article {index}"):

                    st.write(row["article"])

                    if "label" in selected_articles.columns:

                        if row["label"] == 0:

                            st.error("Dataset Label: Fake News")

                        elif row["label"] == 1:

                            st.success("Dataset Label: Real News")

                    st.caption(
                        f"Cluster: {int(row['cluster'])}"
                    )

            st.caption(
                "Articles shown here are sampled from the clustering results. "
                "Dataset labels are shown for reference and were not used "
                "by KMeans."
            )

    except FileNotFoundError:

        st.error(
            "Clustering results file was not found. "
            "Please generate news_clustering_results.csv "
            "from main.ipynb."
        )

    except Exception as e:

        st.error(
            f"Unable to display clustering results: {e}"
        )


# ==========================================
# 10. ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.write(
        "Fake News Classification and News Article Clustering "
        "using Machine Learning."
    )

    st.divider()

    st.subheader("Dataset")

    st.write("""
    This project uses the Kaggle ISOT Fake and Real News Dataset.

    - Fake.csv: Fake news articles.
    - True.csv: Real news articles.
    - Final cleaned dataset: 44,680 articles.
    """)

    st.subheader("Machine Learning Algorithms")

    st.markdown("""
    **Classification**
    - TF-IDF Vectorization
    - Logistic Regression
    - Multinomial Naive Bayes
    - Linear Support Vector Classification (LinearSVC)

    **Clustering**
    - TF-IDF Vectorization
    - MiniBatch KMeans
    - Truncated SVD for dimensionality reduction and visualization
    """)

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("SVM Accuracy", "99.53%")
    col2.metric("Number of Clusters", "14")

    st.write(
        "The highest tested silhouette score for clustering "
        "was 0.0263 at K = 14."
    )

    st.warning(
        "The clustering silhouette score is low, indicating "
        "substantial overlap between the generated clusters. "
        "The classification accuracy is measured on the project's "
        "held-out test split."
    )

    st.divider()

    st.caption(
        "Developed as a Machine Learning academic project."
    )
