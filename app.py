import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
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
# LOAD FILES
# --------------------------------------------------
@st.cache_resource
def load_data():

    required_files = [
        "popularity.pkl",
        "cf_data.pkl",
        "user_sim_df.pkl",
        "user_book_matrix.pkl",
        "books_cb.pkl",
        "title_to_idx.pkl",
        "tfidf_matrix.pkl"
    ]

    missing = [
        file
        for file in required_files
        if not os.path.exists(file)
    ]

    if missing:
        st.error(
            f"Missing files: {', '.join(missing)}"
        )
        st.stop()

    popularity_model = pickle.load(
        open("popularity.pkl", "rb")
    )

    cf_data = pickle.load(
        open("cf_data.pkl", "rb")
    )

    user_sim_df = pickle.load(
        open("user_sim_df.pkl", "rb")
    )

    user_book_matrix = pickle.load(
        open("user_book_matrix.pkl", "rb")
    )

    books_cb = pickle.load(
        open("books_cb.pkl", "rb")
    )

    title_to_idx = pickle.load(
        open("title_to_idx.pkl", "rb")
    )

    tfidf_matrix = pickle.load(
        open("tfidf_matrix.pkl", "rb")
    )

    return (
        popularity_model,
        cf_data,
        user_sim_df,
        user_book_matrix,
        books_cb,
        title_to_idx,
        tfidf_matrix
    )


(
    popularity_model,
    cf_data,
    user_sim_df,
    user_book_matrix,
    books_cb,
    title_to_idx,
    tfidf_matrix
) = load_data()

# --------------------------------------------------
# RECOMMEND USER
# --------------------------------------------------
def recommend_for_user(
    user_id,
    n_recommendations=10
):

    try:

        similar_users = (
            user_sim_df[user_id]
            .drop(user_id)
            .sort_values(
                ascending=False
            )
            .head(10)
            .index
        )

        user_books = set(
            cf_data[
                cf_data["User-ID"]
                == user_id
            ]["Book-Title"]
        )

        recommendations = {}

        for sim_user in similar_users:

            sim_ratings = cf_data[
                (
                    cf_data["User-ID"]
                    == sim_user
                )
                &
                (
                    cf_data["Book-Rating"]
                    >= 7
                )
            ]

            for _, row in sim_ratings.iterrows():

                book = row["Book-Title"]

                if book not in user_books:

                    recommendations.setdefault(
                        book,
                        []
                    ).append(
                        row["Book-Rating"]
                    )

        rec_df = pd.DataFrame([
            {
                "Book Title": book,
                "Predicted Rating":
                round(
                    np.mean(ratings),
                    2
                )
            }
            for book, ratings
            in recommendations.items()
        ])

        return rec_df.sort_values(
            "Predicted Rating",
            ascending=False
        ).head(n_recommendations)

    except:
        return pd.DataFrame()

# --------------------------------------------------
# CONTENT BASED
# --------------------------------------------------
def recommend_similar_books(
    book_title,
    n=10
):

    try:

        search = book_title.lower()

        matches = [
            title
            for title
            in title_to_idx.index
            if search in title
        ]

        if len(matches) == 0:
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

        result[
            "Similarity Score"
        ] = np.round(
            similarity[top_indices],
            3
        )

        return result.reset_index(
            drop=True
        )

    except:
        return pd.DataFrame()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 EDA Dashboard",
        "🔍 Search Books",
        "⭐ Popularity-Based",
        "👥 Collaborative Filtering",
        "📖 Content-Based"
    ]
)

# --------------------------------------------------
# HOME
# --------------------------------------------------
if menu == "🏠 Home":

    st.title(
        "📚 Book Recommendation System"
    )

    st.markdown(
        """
        Welcome to the Book Recommendation System.

        This application provides:

        - ⭐ Popularity-Based Recommendations
        - 👥 Collaborative Filtering
        - 📖 Content-Based Recommendations
        - 📊 Exploratory Data Analysis
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Books",
        books_cb["Book-Title"]
        .nunique()
    )

    col2.metric(
        "Users",
        cf_data["User-ID"]
        .nunique()
    )

    col3.metric(
        "Ratings",
        len(cf_data)
    )

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif menu == "📊 EDA Dashboard":

    st.title(
        "📊 Exploratory Data Analysis"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Books",
        books_cb["Book-Title"]
        .nunique()
    )

    col2.metric(
        "Authors",
        books_cb["Book-Author"]
        .nunique()
    )

    col3.metric(
        "Users",
        cf_data["User-ID"]
        .nunique()
    )

    st.subheader(
        "Top Rated Books"
    )

    st.dataframe(
        popularity_model.head(10),
        use_container_width=True
    )

    st.subheader(
        "Rating Distribution"
    )

    fig, ax = plt.subplots()

    cf_data["Book-Rating"].hist(
        bins=10,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader(
        "Top Authors"
    )

    top_authors = (
        books_cb["Book-Author"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        top_authors
    )

# --------------------------------------------------
# SEARCH
# --------------------------------------------------
elif menu == "🔍 Search Books":

    st.title(
        "🔍 Search Books"
    )

    search = st.text_input(
        "Enter Book Name"
    )

    if search:

        results = books_cb[
            books_cb["Book-Title"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            results.head(50),
            use_container_width=True
        )

# --------------------------------------------------
# POPULARITY
# --------------------------------------------------
elif menu == "⭐ Popularity-Based":

    st.title(
        "⭐ Popular Books"
    )

    st.dataframe(
        popularity_model[
            [
                "Book-Title",
                "Book-Author",
                "avg_rating",
                "num_ratings"
            ]
        ].head(20),
        use_container_width=True
    )

# --------------------------------------------------
# COLLABORATIVE
# --------------------------------------------------
elif menu == "👥 Collaborative Filtering":

    st.title(
        "👥 User-Based Recommendation"
    )

    user_ids = sorted(
        cf_data["User-ID"]
        .unique()
        .tolist()
    )

    selected_user = st.selectbox(
        "Select User",
        user_ids
    )

    st.subheader(
        "User History"
    )

    history = cf_data[
        cf_data["User-ID"]
        == selected_user
    ]

    st.dataframe(
        history[
            [
                "Book-Title",
                "Book-Rating"
            ]
        ],
        use_container_width=True
    )

    if st.button(
        "Recommend Books"
    ):

        recs = recommend_for_user(
            selected_user
        )

        if recs.empty:
            st.warning(
                "No recommendations found."
            )
        else:
            st.dataframe(
                recs,
                use_container_width=True
            )

# --------------------------------------------------
# CONTENT BASED
# --------------------------------------------------
elif menu == "📖 Content-Based":

    st.title(
        "📖 Similar Books"
    )

    books = sorted(
        books_cb["Book-Title"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_book = st.selectbox(
        "Select Book",
        books
    )

    if st.button(
        "Find Similar Books"
    ):

        recs = recommend_similar_books(
            selected_book
        )

        if recs.empty:
            st.warning(
                "No recommendations found."
            )
        else:
            st.dataframe(
                recs,
                use_container_width=True
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Book Recommendation System | Streamlit Deployment"
)
