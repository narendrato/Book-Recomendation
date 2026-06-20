import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():

    required_files = [
        "popularity.pkl",
        "user_sim_df.pkl",
        "user_book_matrix.pkl",
        "cf_data.pkl",
        "books_cb.pkl",
        "tfidf_matrix.pkl",
        "title_to_idx.pkl"
    ]

    missing_files = [
        file for file in required_files
        if not os.path.exists(file)
    ]

    if missing_files:
        st.error(
            f"Missing files: {', '.join(missing_files)}"
        )
        st.stop()

    popularity_model = pickle.load(
        open("popularity.pkl", "rb")
    )

    user_sim_df = pickle.load(
        open("user_sim_df.pkl", "rb")
    )

    user_book_matrix = pickle.load(
        open("user_book_matrix.pkl", "rb")
    )

    cf_data = pickle.load(
        open("cf_data.pkl", "rb")
    )

    books_cb = pickle.load(
        open("books_cb.pkl", "rb")
    )

    tfidf_matrix = pickle.load(
        open("tfidf_matrix.pkl", "rb")
    )

    title_to_idx = pickle.load(
        open("title_to_idx.pkl", "rb")
    )

    return (
        popularity_model,
        user_sim_df,
        user_book_matrix,
        cf_data,
        books_cb,
        tfidf_matrix,
        title_to_idx
    )


(
    popularity_model,
    user_sim_df,
    user_book_matrix,
    cf_data,
    books_cb,
    tfidf_matrix,
    title_to_idx
) = load_models()

# --------------------------------------------------
# COLLABORATIVE FILTERING
# --------------------------------------------------
def recommend_for_user(
    user_id,
    n_recommendations=10
):

    if user_id not in user_sim_df.index:
        return pd.DataFrame()

    similar_users = (
        user_sim_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    user_books = set(
        cf_data[
            cf_data["User-ID"] == user_id
        ]["Book-Title"]
    )

    recommendations = {}

    for sim_user in similar_users:

        sim_ratings = cf_data[
            (cf_data["User-ID"] == sim_user)
            &
            (cf_data["Book-Rating"] >= 7)
        ]

        for _, row in sim_ratings.iterrows():

            title = row["Book-Title"]

            if title not in user_books:

                recommendations.setdefault(
                    title,
                    []
                ).append(
                    row["Book-Rating"]
                )

    if not recommendations:
        return pd.DataFrame()

    rec_df = pd.DataFrame([
        {
            "Book Title": title,
            "Predicted Rating":
            round(np.mean(ratings), 2)
        }
        for title, ratings
        in recommendations.items()
    ])

    rec_df = rec_df.sort_values(
        "Predicted Rating",
        ascending=False
    )

    return rec_df.head(
        n_recommendations
    )

# --------------------------------------------------
# CONTENT-BASED FILTERING
# --------------------------------------------------
def recommend_similar_books(
    book_title,
    n=10
):

    search = book_title.lower()

    matches = [
        title
        for title in title_to_idx.index
        if search in title
    ]

    if not matches:
        return pd.DataFrame()

    idx = title_to_idx[matches[0]]

    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    similarity = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = (
        np.argsort(similarity)[::-1]
        [1:n+1]
    )

    result = books_cb.iloc[
        top_indices
    ][
        [
            "Book-Title",
            "Book-Author"
        ]
    ].copy()

    result["Similarity Score"] = np.round(
        similarity[top_indices],
        3
    )

    return result.reset_index(
        drop=True
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📚 Book Recommendation System")

st.markdown(
    """
    Recommend books using:

    ⭐ Popularity-Based Filtering

    👥 Collaborative Filtering

    📖 Content-Based Filtering
    """
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
model_choice = st.sidebar.radio(
    "Select Recommendation Model",
    [
        "Popularity-Based",
        "Collaborative Filtering",
        "Content-Based"
    ]
)

# --------------------------------------------------
# POPULARITY
# --------------------------------------------------
if model_choice == "Popularity-Based":

    st.header(
        "⭐ Top Popular Books"
    )

    st.dataframe(
        popularity_model[
            [
                "Book-Title",
                "Book-Author",
                "avg_rating",
                "num_ratings"
            ]
        ].head(10),
        use_container_width=True
    )

# --------------------------------------------------
# COLLABORATIVE
# --------------------------------------------------
elif model_choice == "Collaborative Filtering":

    st.header(
        "👥 User-Based Recommendations"
    )

    user_ids = sorted(
        cf_data["User-ID"]
        .unique()
        .tolist()
    )

    selected_user = st.selectbox(
        "Select User ID",
        user_ids
    )

    if st.button(
        "Get Recommendations"
    ):

        recs = recommend_for_user(
            selected_user
        )

        if recs.empty:

            st.warning(
                "No recommendations found."
            )

        else:

            st.success(
                f"Recommendations for User {selected_user}"
            )

            st.dataframe(
                recs,
                use_container_width=True
            )

# --------------------------------------------------
# CONTENT BASED
# --------------------------------------------------
elif model_choice == "Content-Based":

    st.header(
        "📖 Similar Books"
    )

    book_input = st.text_input(
        "Enter Book Title"
    )

    if st.button(
        "Find Similar Books"
    ):

        if not book_input.strip():

            st.warning(
                "Enter a book title."
            )

        else:

            recs = recommend_similar_books(
                book_input
            )

            if recs.empty:

                st.error(
                    "No matching book found."
                )

            else:

                st.success(
                    f"Books similar to '{book_input}'"
                )

                st.dataframe(
                    recs,
                    use_container_width=True
                )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Book Recommendation System using Machine Learning"
)
