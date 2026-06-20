import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
@st.cache_resource
def load_models():

    popularity_model = pickle.load(open("popularity.pkl", "rb"))

    user_sim_df = pickle.load(open("user_sim_df.pkl", "rb"))
    user_book_matrix = pickle.load(open("user_book_matrix.pkl", "rb"))
    filtered_df = pickle.load(open("filtered_df.pkl", "rb"))

    books_cb = pickle.load(open("books_cb.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    title_to_idx = pickle.load(open("title_to_idx.pkl", "rb"))

    return (
        popularity_model,
        user_sim_df,
        user_book_matrix,
        filtered_df,
        books_cb,
        tfidf_matrix,
        title_to_idx
    )


(
    popularity_model,
    user_sim_df,
    user_book_matrix,
    filtered_df,
    books_cb,
    tfidf_matrix,
    title_to_idx
) = load_models()

# --------------------------------------------------
# Collaborative Filtering Function
# --------------------------------------------------
def recommend_for_user(user_id, n_recommendations=10):

    if user_id not in user_sim_df.index:
        return pd.DataFrame()

    similar_users = (
        user_sim_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    rated_books = set(
        filtered_df[
            filtered_df["User-ID"] == user_id
        ]["Book-Title"]
    )

    recommendations = {}

    for sim_user in similar_users:

        sim_ratings = filtered_df[
            (filtered_df["User-ID"] == sim_user)
            & (filtered_df["Book-Rating"] >= 7)
        ]

        for _, row in sim_ratings.iterrows():

            book = row["Book-Title"]

            if book not in rated_books:

                if book not in recommendations:
                    recommendations[book] = []

                recommendations[book].append(
                    row["Book-Rating"]
                )

    if len(recommendations) == 0:
        return pd.DataFrame()

    rec_df = pd.DataFrame(
        [
            {
                "Book-Title": book,
                "Predicted Rating": round(
                    np.mean(ratings), 2
                ),
            }
            for book, ratings in recommendations.items()
        ]
    )

    rec_df = rec_df.sort_values(
        "Predicted Rating",
        ascending=False
    ).head(n_recommendations)

    return rec_df


# --------------------------------------------------
# Content Based Function
# --------------------------------------------------
def recommend_similar_books(book_title, n=10):

    search_title = book_title.lower()

    matches = [
        title
        for title in title_to_idx.index
        if search_title in title
    ]

    if len(matches) == 0:
        return pd.DataFrame()

    idx = title_to_idx[matches[0]]

    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = (
        np.argsort(similarity_scores)[::-1][1:n+1]
    )

    result = books_cb.iloc[top_indices][
        ["Book-Title", "Book-Author"]
    ].copy()

    result["Similarity Score"] = np.round(
        similarity_scores[top_indices],
        3
    )

    return result.reset_index(drop=True)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📚 Book Recommendation System")

st.markdown(
    """
    This application recommends books using:

    - ⭐ Popularity-Based Filtering
    - 👥 Collaborative Filtering
    - 📖 Content-Based Filtering
    """
)

# Sidebar
model_choice = st.sidebar.radio(
    "Choose Recommendation Method",
    [
        "Popularity-Based",
        "Collaborative Filtering",
        "Content-Based"
    ]
)

# --------------------------------------------------
# Popularity Model
# --------------------------------------------------
if model_choice == "Popularity-Based":

    st.header("⭐ Top Popular Books")

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
# Collaborative Filtering
# --------------------------------------------------
elif model_choice == "Collaborative Filtering":

    st.header("👥 User-Based Recommendations")

    user_ids = sorted(
        filtered_df["User-ID"]
        .unique()
        .tolist()
    )

    selected_user = st.selectbox(
        "Select User ID",
        user_ids
    )

    if st.button("Get Recommendations"):

        recommendations = recommend_for_user(
            selected_user
        )

        if recommendations.empty:

            st.warning(
                "No recommendations found."
            )

        else:

            st.success(
                f"Top recommendations for User {selected_user}"
            )

            st.dataframe(
                recommendations,
                use_container_width=True
            )

# --------------------------------------------------
# Content Based
# --------------------------------------------------
else:

    st.header("📖 Similar Books")

    book_input = st.text_input(
        "Enter Book Title"
    )

    if st.button("Find Similar Books"):

        if not book_input.strip():

            st.warning(
                "Please enter a book title."
            )

        else:

            recommendations = recommend_similar_books(
                book_input
            )

            if recommendations.empty:

                st.error(
                    "Book not found."
                )

            else:

                st.success(
                    f"Books similar to '{book_input}'"
                )

                st.dataframe(
                    recommendations,
                    use_container_width=True
                )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Book Recommendation System using Popularity, Collaborative, and Content-Based Filtering."
)