# dashboard.py
import pandas as pd
import numpy as np
import requests
import gradio as gr

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/recommend"

# Load CSV for category dropdowns and thumbnails
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "missing.png",
    books["thumbnail"] + "&fife=w800"
)

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# --- Function to request recommendations from backend API ---
def recommend_books(query, category, tone):
    payload = {"query": query, "category": category, "tone": tone}
    print("\n--- Sending request to backend ---")
    print("Payload:", payload)

    try:
        response = requests.post(API_URL, json=payload)
        print("Response status code:", response.status_code)
        print("Response text (first 300 chars):", response.text[:300])
    except Exception as e:
        print("⚠️ Error connecting to API:", e)
        return [("missing.png", "Cannot connect to backend. Check if FastAPI is running.")]

    if response.status_code != 200:
        return [("missing.png", f"Server error ({response.status_code}). Check terminal for details.")]

    try:
        recs = response.json()
    except Exception as e:
        print("⚠️ Error parsing JSON:", e)
        return [("missing.png", "Invalid response from backend.")]

    # Convert API response to display format
    results = []
    for row in recs:
        desc = row.get("description", "")
        truncated_description = " ".join(desc.split()[:30]) + "..."
        authors = row.get("authors", "Unknown")
        caption = f"{row.get('title', 'Untitled')} by {authors}: {truncated_description}"
        results.append((row.get("large_thumbnail", "missing.png"), caption))

    return results


# --- Build Gradio dashboard ---
with gr.Blocks(theme=gr.themes.Ocean()) as dashboard:
    gr.Markdown("# 📚 Book Recommender Dashboard")
    gr.Markdown("Describe what kind of book you want and get recommendations based on meaning and emotion.")

    with gr.Row():
        user_query = gr.Textbox(label="Describe the kind of book", placeholder="e.g., a book about war")
        category_dropdown = gr.Dropdown(choices=categories, label="Select category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select tone", value="All")
        submit_button = gr.Button("Search Recommendation")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# --- Launch dashboard ---
if __name__ == "__main__":
    dashboard.launch(share=True)
