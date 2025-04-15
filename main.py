import re
import customtkinter
import numpy as np
import pandas as pd
from tkinter import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#importing data set
movie_dataset = pd.read_csv("movies.csv")
ratings_dataset = pd.read_csv("ratings.csv")


# cleaning movie titles
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movie_dataset["clean_title"] = movie_dataset["title"].apply(clean_title)

#Matrix
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movie_dataset["clean_title"])

# search function
def search():
    title = titleEntry.get()
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec,tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movie_dataset.iloc[indices][::-1]
    return results


# recommendation system
def similar_movies(movie_id):
    #finds users similar to us
    sim_users = ratings_dataset[(ratings_dataset["movieId"] == movie_id) & (ratings_dataset["rating"] >= 4)][
        "userId"].unique()
    sim_user_recs = ratings_dataset[(ratings_dataset["userId"].isin(sim_users)) & (ratings_dataset["rating"] >= 4)][
        "movieId"]
    #adjusting recommendations so only has recs where over 10% of users recommended that movie
    sim_user_recs = sim_user_recs.value_counts() / len(sim_users)
    sim_user_recs = sim_user_recs[sim_user_recs > .1]

    #finding how common all of recs were along all users
    all_users = ratings_dataset[
        (ratings_dataset["movieId"].isin(sim_user_recs.index)) & (ratings_dataset["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    #creating score
    rec_percentages = pd.concat([sim_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    topten = rec_percentages.head(10).merge(movie_dataset, left_index=True, right_on="movieId")
    return topten[["title","genres","score"]]

#gui

# appearance
customtkinter.set_appearance_mode("System")  # light / dark / system
customtkinter.set_default_color_theme("blue")

# root
root = customtkinter.CTk()
root.title("Movie Recommendation System")
root.geometry("1000x500")

# frame to hold entry and button side by side
input_frame = customtkinter.CTkFrame(root, fg_color="transparent")
input_frame.pack(pady=50)

# entry field
titleEntry = customtkinter.CTkEntry(input_frame, placeholder_text="Enter Movie Title: ", width=300)
titleEntry.pack(side="left", padx=10)

# button
button = customtkinter.CTkButton(input_frame, text="Submit", command=lambda: searchclick())
button.pack(side="left")

# textbox that will display output - initialized blank
mytext = customtkinter.CTkTextbox(root, width=650, height=250, font=("Courier New", 12)
                                  ,pady=20, padx=20)
mytext.pack(pady=10)
mytext.configure(state="disabled")  # Make it read-only until we update it

# search function trigger
def searchclick():
    results = search()
    movie_id = results.iloc[0]["movieId"]
    top_recs = similar_movies(movie_id)

    # Format the data into a nice clean table
    header = f"{'Title':<35} {'Genres':<40} {'Score':>6}\n"
    separator = "-" * 85 + "\n"
    rows = ""

    for _, row in top_recs.iterrows():
        title = row['title'][:33] + "…" if len(row['title']) > 35 else row['title']
        genres = row['genres'].replace("|", ", ")
        genres = genres[:38] + "…" if len(genres) > 40 else genres
        score = f"{row['score']:.2f}"
        rows += f"{title:<35} {genres:<40} {score:>6}\n"

    table_text = header + separator + rows

    # Step 3: Update the existing textbox
    mytext.configure(state="normal")        # Unlock it
    mytext.delete("0.0", "end")             # Clear old content
    mytext.insert("0.0", table_text)        # Insert new content
    mytext.configure(state="disabled")      # Lock it again

root.mainloop()