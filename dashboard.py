#we will use gradio, a python package to showcase machine learning models

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "missing.png",
    books["large_thumbnail"],
)
#-EXPLANATION FROM ABOVE-
#the dataset used provide a url linked to google books but book covers that kind of random sizes
#but we want the books to return largest possible size available to get better resolution
#there are numbers of books that don't have covers, if we try to render them we will get error, thus in the repo we add
#an image for showing cover not found

raw_documents = TextLoader("tag_description5.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
#This section is to build vector database to do the core functionality of this recommender
#which is semantic recommendation

#now create a function to retrieve semantic recommendation and also going to apply filtering based on category
# and sorting based on emotional tone

def retrieve_semantic_recommendation(
        query:str,
        category:str = None,
        tone: str = None,
        initial_top_k = 50,
        final_top_k = 16
) -> pd.DataFrame:                   #the result return is in pandas dataframe
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    #books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_list = [rec.page_content.strip('"').split()[0] for rec, _ in recs]
    book_recs = books[books["isbn10"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # now applying filtering based on categories, its going to be a drop down
    if tone == "Happy":
        book_recs.sort_values(by=["joy"], ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by=["surprise"], ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by=["anger"], ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by=["fear"], ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by=["sadness"], ascending=False, inplace=True)

    return book_recs

# a function for specified what we need to display on the dashboard
def recommend_books (
        query:str,
        category:str,
        tone:str
):
    recommendation = retrieve_semantic_recommendation(query, category, tone)
    results = []
    for _, row in recommendation.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."    #the dashboard has limited space, don't show full desc
                                    #the description is splitted into seperated words, if more than 30 cut it of

        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            author_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            author_str = f"{', '.join(author_split[:-1])} and {author_split[-1]}"
        else :
            author_str = row["authors"]

        #now to display all of them is as the caption displayed on the bottom thumbnail
        caption = f"{row['title']} by {author_str}: {truncated_description}" #combaine them into a caption str
        results.append((row["large_thumbnail"], caption)) #we append a tupple containing the thumbnail and the caption
    return results

#building dashboard
categories = ["All"] + sorted(books["simple_categories"].unique())          #list containing categories
tones =  ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]   #List containing tones

with gr.Blocks(theme= gr.themes.Ocean()) as dashboard:     #check out https://www.gradio.app/guides/theming-guide
    gr.Markdown("# Book Recommender")
    with gr.Row():
        user_query = gr.Textbox(label = "please describe the kind of book",
                                placeholder = "e.g., a book about war" )
        category_dropdown = gr.Dropdown(choices = categories, label = "select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "select a tone:", value = "All")
        submit_button = gr.Button("search recommendation")

    gr.Markdown("## Recommendation")
    output = gr.Gallery(label= "recommended books", columns = 8, rows = 2)  #match the final top_k which is 16

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

#main method
if __name__ == "__main__":
    dashboard.launch()