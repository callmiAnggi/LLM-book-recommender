# backend/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
app = FastAPI()

# Load data once at startup
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("missing.png") + "&fife=w800"

# Build Chroma index
raw_documents = TextLoader("tag_description5.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# Request model
class QueryRequest(BaseModel):
    query: str
    category: str
    tone: str

@app.post("/recommend")
async def recommend_books(req: QueryRequest):
    query, category, tone = req.query, req.category, req.tone
    recs = db_books.similarity_search_with_score(query, k=50)
    books_list = [rec.page_content.strip('"').split()[0] for rec, _ in recs]
    book_recs = books[books["isbn10"].isin(books_list)].head(16)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(16)

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

    # return the top books as JSON
    return book_recs.to_dict(orient="records")
