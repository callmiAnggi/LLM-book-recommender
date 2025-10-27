# app.py
import os
import time
import pandas as pd
import numpy as np
import gradio as gr
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from gradio.routes import mount_gradio_app

load_dotenv()
print("🔑 OpenAI key detected:", os.getenv("OPENAI_API_KEY") is not None)
app = FastAPI()

# ============ CORS ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LOAD DATA ============
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "missing.png",
    books["thumbnail"] + "&fife=w800"
)

# ============ BUILD EMBEDDINGS ============
raw_documents = TextLoader("tag_description5.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# ============ RATE LIMIT ============
rate_limit_store = {}
MAX_REQUESTS = 2
WINDOW_SECONDS = 86400

def rate_limit(ip: str):
    current_time = time.time()
    entry = rate_limit_store.get(ip)
    if entry:
        count, first_time = entry
        if current_time - first_time > WINDOW_SECONDS:
            rate_limit_store[ip] = (1, current_time)
        elif count >= MAX_REQUESTS:
            raise HTTPException(status_code=429, detail="You have reached the maximum of 2 free requests.")
        else:
            rate_limit_store[ip] = (count + 1, first_time)
    else:
        rate_limit_store[ip] = (1, current_time)

# ============ API ENDPOINT ============
@app.post("/recommend")
async def recommend(request: Request):
    client_ip = request.client.host
    rate_limit(client_ip)
    data = await request.json()
    query = data.get("query", "")
    category = data.get("category", "All")
    tone = data.get("tone", "All")

    recs = db_books.similarity_search_with_score(query, k=50)
    books_list = [rec.page_content.strip('"').split()[0] for rec, _ in recs]
    book_recs = books[books["isbn10"].isin(books_list)].head(16)

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

    results = book_recs.replace({np.nan: None}).to_dict(orient="records")
    return results


# ============ GRADIO FRONTEND ============
API_URL = "/recommend"
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

def recommend_books(query, category, tone):
    import requests
    payload = {"query": query, "category": category, "tone": tone}
    response = requests.post(f"http://127.0.0.1:7860{API_URL}", json=payload)
    if response.status_code != 200:
        return [("missing.png", "Server error, please try again.")]
    recs = response.json()
    results = []
    for row in recs:
        desc = row.get("description", "")
        truncated = " ".join(desc.split()[:30]) + "..."
        authors = row.get("authors", "Unknown")
        caption = f"{row['title']} by {authors}: {truncated}"
        results.append((row.get("large_thumbnail", "missing.png"), caption))
    return results

with gr.Blocks(theme=gr.themes.Ocean()) as dashboard:
    gr.Markdown("# 📚 Book Recommender")
    with gr.Row():
        user_query = gr.Textbox(label="Describe the kind of book")
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Tone", value="All")
        submit_button = gr.Button("Search Recommendation")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=2)
    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

# Mount Gradio inside FastAPI
app = mount_gradio_app(app, dashboard, path="/")

