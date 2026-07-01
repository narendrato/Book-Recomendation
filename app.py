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
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.book-title {
    font-size: 14px;
    font-weight: bold;
    min-height: 60px;
    color: white;
}

.book-author {
    color: #BBBBBB;
    font-size: 12px;
}

.book-rating {
    color: gold;
    font-size: 13px;
}

img {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA
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
        f for f in required_files
        if not os.path.exists(f)
    ]

    if missing:
        st.error(f"Missing files: {missing}")
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
# BOOK COVER
# --------------------------------------------------
def get_book_cover(isbn):

    if pd.isna(isbn):
        return "https://via.placeholder.com/150x220?text=No+Cover"

    return (
        f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    )


def get_isbn(book_title):

    data = books_cb[
        books_cb["Book-Title"] == book_title
    ]

    if len(data):
        return data.iloc[0]["ISBN"]

    return None


# --------------------------------------------------
# DISPLAY BOOK CARDS
# --------------------------------------------------
def show_book_cards(df, books_per_row=5):

    for i in range(
        0,
        len(df),
        books_per_row
    ):

        cols = st.columns(books_per_row)

        for j, col in enumerate(cols):

            if i + j < len(df):

                book = df.iloc[i + j]

                isbn = get_isbn(
                    book["Book-Title"]
                )

                with col:

                    st.image(
                        get_book_cover(isbn),
                        use_container_width=True
                    )

                    st.markdown(
                        f"""
                        <div class='book-title'>
                        {book['Book-Title']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class='book-author'>
                        {book['Book-Author']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if "avg_rating" in book:

                        st.markdown(
                            f"""
                            <div class='book-rating'>
                            ⭐ {round(book['avg_rating'],2)}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


# --------------------------------------------------
# COLLABORATIVE FILTERING
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
                cf_data["User-ID"] == user_id
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
                "Book-Title": book,
                "Book-Author":
                books_cb[
                    books_cb["Book-Title"]
                    == book
                ]["Book-Author"].iloc[0]
                if len(
                    books_cb[
                        books_cb["Book-Title"]
                        == book
                    ]
                )
                else "Unknown",

                "avg_rating":
                round(
                    np.mean(ratings),
                    2
                )
            }

            for book, ratings
            in recommendations.items()
        ])

        return rec_df.sort_values(
            "avg_rating",
            ascending=False
        ).head(n_recommendations
            except Exception as e:

        print(e)
        return pd.DataFrame()
# --------------------------------------------------
# CONTENT-BASED RECOMMENDATIONS
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
                cf_data["User-ID"] == user_id
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
                "Book-Title": book,
                "Book-Author":
                books_cb[
                    books_cb["Book-Title"] == book
                ]["Book-Author"].iloc[0]
                if len(
                    books_cb[
                        books_cb["Book-Title"] == book
                    ]
                )
                else "Unknown",

                "avg_rating":
                round(
                    np.mean(ratings),
                    2
                )
            }

            for book, ratings
            in recommendations.items()
        ])

        return rec_df.sort_values(
            "avg_rating",
            ascending=False
        ).head(n_recommendations)

    except Exception as e:

        st.error(
            f"Recommendation Error: {e}"
        )

        return pd.DataFrame()
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
menu = st.sidebar.radio(
    "📌 Navigation",
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

        ### Features

        - ⭐ Popularity-Based Recommendations
        - 👥 Collaborative Filtering
        - 📖 Content-Based Recommendations
        - 📊 Exploratory Data Analysis
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Books",
        books_cb["Book-Title"].nunique()
    )

    col2.metric(
        "Users",
        cf_data["User-ID"].nunique()
    )

    col3.metric(
        "Ratings",
        len(cf_data)
    )

    st.markdown("---")

    st.subheader("🔥 Trending Books")

    show_book_cards(
        popularity_model.head(10)
    )


# --------------------------------------------------
# EDA DASHBOARD
# --------------------------------------------------
elif menu == "📊 EDA Dashboard":

    st.title(
        "📊 Exploratory Data Analysis"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Books",
        books_cb["Book-Title"].nunique()
    )

    col2.metric(
        "Authors",
        books_cb["Book-Author"].nunique()
    )

    col3.metric(
        "Users",
        cf_data["User-ID"].nunique()
    )

    st.markdown("---")

    st.subheader(
        "🏆 Top Rated Books"
    )

    st.dataframe(
        popularity_model.head(10),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "📈 Rating Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    cf_data[
        "Book-Rating"
    ].hist(
        bins=10,
        ax=ax
    )

    ax.set_xlabel(
        "Rating"
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader(
        "✍️ Top Authors"
    )

    top_authors = (
        books_cb[
            "Book-Author"
        ]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        top_authors
    )


# --------------------------------------------------
# SEARCH BOOKS
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
            books_cb[
                "Book-Title"
            ].str.contains(
                search,
                case=False,
                na=False
            )
        ]

        if results.empty:

            st.warning(
                "No books found."
            )

        else:

            show_book_cards(
                results.head(20)
            )
            # --------------------------------------------------
# POPULARITY BASED
# --------------------------------------------------
elif menu == "⭐ Popularity-Based":

    st.title("⭐ Popular Books")

    st.markdown(
        "Top books based on average ratings and number of users."
    )

    st.markdown("---")

    show_book_cards(
        popularity_model.head(20)
    )


# --------------------------------------------------
# COLLABORATIVE FILTERING
# --------------------------------------------------
elif menu == "👥 Collaborative Filtering":

    st.title(
        "👥 Reader Profiles & Recommendations"
    )

    user_summary = (
        cf_data.groupby("User-ID")
        .agg(
            Books_Rated=("Book-Title", "count"),
            Avg_Rating=("Book-Rating", "mean")
        )
        .reset_index()
    )

    user_summary["Avg_Rating"] = (
        user_summary["Avg_Rating"]
        .round(2)
    )

    user_summary["Profile"] = (
        "Reader "
        + user_summary["User-ID"].astype(str)
        + " | Books Rated: "
        + user_summary["Books_Rated"].astype(str)
        + " | Avg Rating: "
        + user_summary["Avg_Rating"].astype(str)
    )

    selected_profile = st.selectbox(
        "Select Reader Profile",
        user_summary["Profile"]
    )

    selected_user = int(
        selected_profile
        .split("|")[0]
        .replace("Reader", "")
        .strip()
    )

    profile = user_summary[
        user_summary["User-ID"]
        == selected_user
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "Books Rated",
        int(
            profile["Books_Rated"]
            .iloc[0]
        )
    )

    col2.metric(
        "Average Rating",
        float(
            profile["Avg_Rating"]
            .iloc[0]
        )
    )

    st.markdown("---")

    st.subheader(
        "📚 Reading History"
    )

    history = cf_data[
        cf_data["User-ID"]
        == selected_user
    ][
        [
            "Book-Title",
            "Book-Rating"
        ]
    ]

    st.dataframe(
        history.sort_values(
            "Book-Rating",
            ascending=False
        ),
        use_container_width=True
    )

    st.markdown("---")

    if st.button(
        "🎯 Get Personalized Recommendations"
    ):

        recommendations = recommend_for_user(
            selected_user
        )

        if recommendations.empty:

            st.warning(
                "No recommendations available."
            )

        else:

            st.subheader(
                "📖 Recommended Books"
            )

            show_book_cards(
                recommendations
            )


# --------------------------------------------------
# CONTENT BASED FILTERING
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

            st.subheader(
                "📚 Similar Books"
            )

            show_book_cards(
                recs
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <center>

    📚 <b>Book Recommendation System</b><br>
    Built with Streamlit, Scikit-Learn & Pandas

    </center>
    """,
    unsafe_allow_html=True
)
    except:

        return pd.DataFrame()
