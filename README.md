# Chatbot_utena

A Streamlit-powered chatbot app for exploring and answering questions about Utena, Lithuania.

## Features
- **Multi-source knowledge:**
  - Web search (custom URL, Wikipedia, or Utena tourism info site as fallback)
  - Text file upload (or default utena.txt)
  - PDF upload (or default Utena-region-overview.pdf)
- **Flexible source selection:**
  - Choose to use Web, Text file, PDF, or All sources together
- **RAG (Retrieval-Augmented Generation):**
  - Uses LangChain, OpenAI embeddings, and Chroma vectorstore for context-aware answers
- **Source transparency:**
  - Shows which sources were used for each answer
- **User-friendly UI:**
  - Upload files, enter URLs, and ask questions directly in the browser

## How it works
1. **Select a knowledge source** in the sidebar (Web, Text file, PDF, or All)
2. **For Web:**
   - Enter a URL or leave blank to use Wikipedia/utena.info as fallback
3. **For Text file:**
   - Upload a `.txt` file or use the default `utena.txt`
4. **For PDF:**
   - Upload a PDF or use the default `Utena-region-overview.pdf`
5. **Ask a question** about Utena in the input box
6. **View the answer and sources** used for the response

## Requirements
- Python 3.9+
- All dependencies are listed in `pyproject.toml` and `requirements.txt`
- Uses `uv`, `pip`, or `venv` for environment management

## Running the app
```sh
uv pip install -r requirements.txt  # or pip install -r requirements.txt
streamlit run chatbot_utena.py
```

## Project structure
- `chatbot_utena.py` — Main Streamlit app
- `utena.txt` — Default text knowledge file
- `Utena-region-overview.pdf` — Default PDF knowledge file
- `requirements.txt`, `pyproject.toml` — Dependencies

## About
This project helps users explore information about Utena, Lithuania, using modern AI and NLP tools. It is ideal for students, tourists, and locals who want quick, source-cited answers about the region.# Chatbot_utena
