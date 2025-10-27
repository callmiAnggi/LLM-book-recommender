# 📚 Book Recommender (Semantic + Emotion-based)
A semantic book recommendation demo using OpenAI embeddings and LangChain, deployed with Gradio on Hugging Face Spaces.

🚀 **Live Demo:** [Try it on Hugging Face](https://anggi99-book-recommender9.hf.space)  

---
### 🧠 Project Overview
This app uses `langchain` + `Chroma` vector search to find books similar to your description, filtered by category and emotional tone.

### ⚙️ Tech Stack
- Python  
- LangChain + ChromaDB  
- OpenAI Embeddings  
- Gradio (UI)  
- Hugging Face Spaces (Deployment)

### 🔒 Note
For this demo, **no backend API** is used — all computations (embeddings and search) run inside the Gradio app directly.  
In production, these would be moved to a backend API with rate limits, secure keys, and caching for performance and cost control.
