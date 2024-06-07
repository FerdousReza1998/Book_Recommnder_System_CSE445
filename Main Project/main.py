import pickle
import streamlit as st
import numpy as np

# Load the required data
model = pickle.load(open('model/KNN_Model.pkl', 'rb'))
books = pickle.load(open('model/book_names.pkl', 'rb'))
final_ratings = pickle.load(open('model/final_ratings.pkl', 'rb'))
df_pivot = pickle.load(open('model/df_pivot.pkl', 'rb'))
popular_df = pickle.load(open('model/popular_df.pkl', 'rb'))


# Function to fetch poster URLs
def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []
    authors = []
    publishers = []
    ratings = []

    for book_id in suggestion[0]:
        book_name.append(df_pivot.index[book_id])
        # df_final = final_ratings[final_ratings['title'] == book_name].drop_duplicates('title')



    for name in book_name: 
        ids = np.where(final_ratings['title'] == name)[0][0]
        
        ids_index.append(ids)

    for idx in ids_index:
        url = final_ratings.iloc[idx]['image_url']
        poster_url.append(url)
        author = final_ratings.iloc[idx]['author']
        authors.append(author)
        rating = final_ratings.iloc[idx]['rating']
        ratings.append(rating)
        publisher = final_ratings.iloc[idx]['publisher']
        publishers.append(publisher)



    return poster_url, authors, publishers, ratings

# Function to recommend books
def recommend_book(book_name):
    books_list = []
    book_id = np.where(df_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(df_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url, authors, publishers, ratings = fetch_poster(suggestion)
    
    for i in range(len(suggestion[0])):
        item = []
        books = df_pivot.index[suggestion[0][i]]

        books_list.append(books)
        
    return books_list, poster_url, authors, publishers, ratings

# Streamlit App
st.header('Book Recommender System Using Machine Learning')

# Sidebar
st.sidebar.title("Menu")
option = st.sidebar.selectbox(
    "Choose a section",
    ["Popular Books", "Recommend Books"]
)

if option == "Popular Books":
    st.subheader("Top 50 Popular Books")
    cols = st.columns(3)
    for idx, col in enumerate(cols):
        with col:
            for i in range(idx, len(popular_df), 3):
                st.image(popular_df['image_url'].values[i], width=100)
                st.write(f"**{popular_df['title'].values[i]}** by {popular_df['author'].values[i]}")
                st.write(f"Votes: {popular_df['num_of_ratings'].values[i]}, Rating: {popular_df['average_rating'].values[i]:.2f}")
                # st.write(f"Rating: {popular_df['average_rating'].values[i]:.2f}")

                st.write("----")

if option == "Recommend Books":
    st.subheader("Recommend Books")
    
    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        books
    )

    if st.button('Show Recommendation'):
        recommended_books, poster_url, authors, publishers, ratings = recommend_book(selected_books)
        cols = st.columns(5)
        for idx, col in enumerate(cols, start=1):
            with col:
                    st.image(poster_url[idx])
                    st.write(f"**{recommended_books[idx]}**")
                    st.write(f"Author: {authors[idx]}")
                    st.write(f"Publisher: {publishers[idx]}")
                    # st.write(f"Rating: {ratings[idx]}")
        
       

            





