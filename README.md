# 🚀 RAG (Retrieval-Augmented Generation) Project with ColBERT

This project implements a robust RAG workflow using the advanced ranking model **ColBERT** to significantly boost retrieval precision and efficiency. ColBERT's "Late Interaction" paradigm is key, allowing for the high accuracy of a cross-encoder while maintaining the scalability required for large document collections.

The included example demonstrates querying a large corpus (Victor Hugo's *Le Dernier Jour d'un Condamné*) to highlight how ColBERT can retrieve highly precise, nuanced passages essential for complex literary analysis.

## ✨ Key Features

* **Efficient ColBERT Retrieval:** Implemented using `RAGPretrainedModel` from Ragatouille.
* **Complete RAG Workflow:** Seamlessly integrates with LangChain for retrieval and a Gemini model (`gemini-2.5-flash`) for final answer generation.
* **Smart Index Management:** The `query` function automatically loads an existing ColBERT index or builds a new one from source documents.
* **Advanced Document Processing:** Includes a `load_and_split` function utilizing a Hugging Face tokenizer (`intfloat/multilingual-e5-small`) for optimized, multilingual document chunking (PDF support).

## 🛠️ Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_LINK]
    cd [PROJECT_NAME]
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies from `requirements.txt`:**

    ```bash
    pip install -r requirements.txt
    ```

    **`requirements.txt` content:**
    ```text
    ragatouille
    langchain-core
    langchain-google-genai
    langchain-community
    pydantic
    transformers
    python-dotenv
    ```

4.  **API Key Configuration:**
    Create a `.env` file in the project root and add your Gemini API key:

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

## ⚙️ Usage Example

The RAG workflow is encapsulated in the `query` function, located (for example) in `rag_colbert.py`.

### 1. Index Building (First Run)

To create a new index (`victorHugoIndex`) from your documents:

```python
# Assuming your source documents (e.g., PDFs) are in './data/hugo/'
DOCS_PATH = "./data/hugo/" 

# Setting index_path=None triggers index creation
answer = query(query_text="Initial test query.", docs_path=DOCS_PATH, index_path=None)
print(f"Index built and first answer generated: {answer}")