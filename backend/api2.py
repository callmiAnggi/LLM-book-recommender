# backend/api.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import time

load_dotenv()

app = FastAPI()

# Allow frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this later to your actual frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "missing.png",
    books["thumbnail"] + "&fife=w800"
)

# Build embeddings
raw_documents = TextLoader("tag_description5.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# =============== RATE LIMIT SECTION ===============
rate_limit_store = {}
MAX_REQUESTS = 2          # max free API uses per IP
WINDOW_SECONDS = 86400    # reset every 24 hours

def rate_limit(ip: str):
    """Simple in-memory rate limiter"""
    current_time = time.time()
    entry = rate_limit_store.get(ip)

    if entry:
        count, first_time = entry
        if current_time - first_time > WINDOW_SECONDS:
            rate_limit_store[ip] = (1, current_time)
        elif count >= MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="You have reached the maximum of 2 free requests. Please contact the developer for access."
            )
        else:
            rate_limit_store[ip] = (count + 1, first_time)
    else:
        rate_limit_store[ip] = (1, current_time)
# ===================================================


@app.post("/recommend")
async def recommend(request: Request):
    client_ip = request.client.host
    rate_limit(client_ip)  # apply limit ✅

    data = await request.json()
    query = data.get("query", "")
    category = data.get("category", "All")
    tone = data.get("tone", "All")

    recs = db_books.similarity_search_with_score(query, k=50)
    books_list = [rec.page_content.strip('"').split()[0] for rec, _ in recs]
    book_recs = books[books["isbn10"].isin(books_list)].head(16)

    # Optional filters
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }
    if tone in tone_map:
        book_recs = book_recs.sort_values(by=[tone_map[tone]], ascending=False)

    # 🧼 Clean NaN/inf before returning
    book_recs = book_recs.replace([np.inf, -np.inf, np.nan], 0)
    results = book_recs.to_dict(orient="records")

    return JSONResponse(content=results)
