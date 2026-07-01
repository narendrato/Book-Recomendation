import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="📚 Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ----------------------------------------------------
# LOAD ALL FILES
# ----------------------------------------------------
@st.cache_resource
def load_data():

    files = [
        "popularity.pkl",
        "cf_data.pkl",
        "user_sim_df.pkl",
        "user_book_matrix.pkl",
        "books_cb.pkl",
        "title_to_idx.pkl",
        "tfidf_matrix.pkl"
    ]

    missing = [f for f in files if not os.path.exists(f)]

    if missing:
        st.error(f"Missing Files : {missing}")
        st.stop()

    popularity_model = pickle.load(open("popularity.pkl","rb"))
    cf_data = pickle.load(open("cf_data.pkl","rb"))
    user_sim_df = pickle.load(open("user_sim_df.pkl","rb"))
    user_book_matrix = pickle.load(open("user_book_matrix.pkl","rb"))
    books_cb = pickle.load(open("books_cb.pkl","rb"))
    title_to_idx = pickle.load(open("title_to_idx.pkl","rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl","rb"))

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
)=load_data()

# ----------------------------------------------------
# COLLABORATIVE FILTERING
# ----------------------------------------------------
def recommend_for_user(user_id,n_recommendations=10):

    try:

        similar_users = (
            user_sim_df[user_id]
            .drop(user_id)
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        user_books = set(
            cf_data[
                cf_data["User-ID"]==user_id
            ]["Book-Title"]
        )

        recommendations={}

        for sim_user in similar_users:

            ratings = cf_data[
                (cf_data["User-ID"]==sim_user) &
                (cf_data["Book-Rating"]>=7)
            ]

            for _,row in ratings.iterrows():

                if row["Book-Title"] not in user_books:

                    recommendations.setdefault(
                        row["Book-Title"],
                        []
                    ).append(
                        row["Book-Rating"]
                    )

        rec_df=pd.DataFrame([

            {
                "Book Title":book,
                "Predicted Rating":round(np.mean(ratings),2)

            }

            for book,ratings in recommendations.items()

        ])

        return rec_df.sort_values(
            "Predicted Rating",
            ascending=False
        ).head(n_recommendations)

    except:
        return pd.DataFrame()

# ----------------------------------------------------
# CONTENT BASED
# ----------------------------------------------------
def recommend_similar_books(book_title,n=10):

    try:

        search = book_title.lower()

        matches = [

            title

            for title in title_to_idx.index

            if search in title

        ]

        if len(matches)==0:
            return pd.DataFrame()

        idx = title_to_idx[matches[0]]

        if isinstance(idx,pd.Series):
            idx = idx.iloc[0]

        similarity = cosine_similarity(
            tfidf_matrix[idx],
            tfidf_matrix
        ).flatten()

        top = np.argsort(similarity)[::-1][1:n+1]

        result = books_cb.iloc[top][
            [
                "Book-Title",
                "Book-Author"
            ]
        ].copy()

        result["Similarity Score"]=np.round(
            similarity[top],
            3
        )

        return result.reset_index(drop=True)

    except:
        return pd.DataFrame()
        # ==========================================================
# SINGLE PAGE DASHBOARD
# ==========================================================

st.title("📚 Book Recommendation System")

st.markdown("""
Welcome to the **Book Recommendation System**.

This dashboard provides

- ⭐ Popularity Based Recommendation
- 👥 Collaborative Filtering
- 📖 Content Based Recommendation
- 📊 Exploratory Data Analysis
""")

st.markdown("---")

# ==========================================================
# OVERVIEW
# ==========================================================

st.subheader("📊 Dataset Overview")

c1,c2,c3,c4=st.columns(4)

c1.metric(
    "Books",
    books_cb["Book-Title"].nunique()
)

c2.metric(
    "Authors",
    books_cb["Book-Author"].nunique()
)

c3.metric(
    "Users",
    cf_data["User-ID"].nunique()
)

c4.metric(
    "Ratings",
    len(cf_data)
)

st.markdown("---")

# ==========================================================
# SEARCH BOOKS
# ==========================================================

st.header("🔍 Search Books")

search=st.text_input(
    "Enter Book Name"
)

if search:

    result=books_cb[
        books_cb["Book-Title"]
        .str.contains(
            search,
            case=False,
            na=False
        )
    ]

    if len(result)==0:

        st.warning(
            "No books found."
        )

    else:

        st.dataframe(
            result.head(50),
            use_container_width=True
        )

st.markdown("---")

# ==========================================================
# POPULAR BOOKS
# ==========================================================

st.header("⭐ Popular Books")

top_n=st.selectbox(
    "Select Top Rated Books",
    [5,10,15,20],
    index=0
)

popular=popularity_model.sort_values(
    "avg_rating",
    ascending=False
).head(top_n)

st.dataframe(

    popular[
        [
            "Book-Title",
            "Book-Author",
            "avg_rating",
            "num_ratings"
        ]
    ],

    use_container_width=True

)

st.markdown("---")

# ==========================================================
# COLLABORATIVE FILTERING
# ==========================================================

st.header("👥 Reader Profiles")

user_summary=(

    cf_data.groupby("User-ID")

    .agg(

        Books_Rated=("Book-Title","count"),

        Avg_Rating=("Book-Rating","mean")

    )

    .reset_index()

)

user_summary["Avg_Rating"]=user_summary[
    "Avg_Rating"
].round(2)

user_summary["Profile"]=(
    "Reader "
    +user_summary["User-ID"].astype(str)
    +" | Books Rated : "
    +user_summary["Books_Rated"].astype(str)
    +" | Avg Rating : "
    +user_summary["Avg_Rating"].astype(str)
)

selected_profile=st.selectbox(

    "Select Reader",

    user_summary["Profile"]

)

selected_user=int(

    selected_profile.split("|")[0]

    .replace("Reader","")

    .strip()

)

profile=user_summary[
    user_summary["User-ID"]==selected_user
]

c1,c2=st.columns(2)

c1.metric(
    "Books Rated",
    int(profile["Books_Rated"])
)

c2.metric(
    "Average Rating",
    float(profile["Avg_Rating"])
)

st.subheader("📚 Reading History")

history=cf_data[
    cf_data["User-ID"]==selected_user
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

if st.button(
    "🎯 Get Personalized Recommendations"
):

    recs=recommend_for_user(
        selected_user
    )

    if recs.empty:

        st.warning(
            "No recommendations found."
        )

    else:

        st.subheader(
            "Recommended Books"
        )

        st.dataframe(
            recs,
            use_container_width=True
        )

st.markdown("---")

# ==========================================================
# CONTENT BASED
# ==========================================================

st.header("📖 Similar Books")

books=sorted(

    books_cb["Book-Title"]

    .dropna()

    .unique()

    .tolist()

)

selected_book=st.selectbox(

    "Choose Book",

    books

)

if st.button(
    "Find Similar Books"
):

    recs=recommend_similar_books(
        selected_book
    )

    if recs.empty:

        st.warning(
            "No recommendations."
        )

    else:

        st.dataframe(
            recs,
            use_container_width=True
        )

st.markdown("---")

# ==========================================================
# EDA
# ==========================================================

st.header("📊 Exploratory Data Analysis")

left,right=st.columns(2)

with left:

    st.subheader(
        "Rating Distribution"
    )

    fig,ax=plt.subplots(
        figsize=(6,4)
    )

    cf_data[
        "Book-Rating"
    ].hist(
        bins=10,
        ax=ax
    )

    ax.set_xlabel("Rating")

    ax.set_ylabel("Frequency")

    st.pyplot(fig)

with right:

    st.subheader(
        "Top Authors"
    )

    authors=(

        books_cb[
            "Book-Author"
        ]

        .value_counts()

        .head(10)

    )

    st.bar_chart(
        authors
    )

st.markdown("---")

st.caption(
    "📚 Book Recommendation System | Popularity • Collaborative • Content Based"
)
