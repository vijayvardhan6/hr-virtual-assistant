# HR Virtual Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that provides employees with instant, accurate answers to **HR-related questions** — including policies, benefits, and company procedures.  
Built using **LangChain**, **Groq API**, **Qdrant**, and **Streamlit**, it combines document intelligence with conversational AI for context-aware HR support.

---

## Features
 
- **Context-Aware Responses** – Retrieves relevant HR content before generating answers 
- **Qdrant Vector Store Integration** – Efficient document similarity search  
- **Streamlit Interface** – Clean, user-friendly chat experience  
- **Extensible Pipelines** – Modular ingestion, retrieval, and LLM layers  

---

## Architecture Overview

### 1. Document Ingestion  
`ingestion_pipeline.py`  
- Loads `.pdf`, `.txt`, and `.md` files  
- Splits into semantic/text chunks using LangChain splitters  
- Converts chunks into vector embeddings  
- Stores embeddings in **Qdrant**  

```
Documents → Chunking → Embeddings → Qdrant Vector DB
```

---

### 2. Query + Retrieval Flow  
`retrieval_pipeline.py`  
- User query → Embedded into vector space  
- Similarity search against stored HR chunks  
- Optionally re-ranked via cross-encoder  
- Context assembled for LLM prompt  

```
User Query → Embedding → Qdrant Search → Top-k Results → Context
```

---

### 3. Response Generation  
`llm.py`  
- Uses **Groq API** (or any LangChain-compatible model)  
- Integrates history-aware prompt handling  
- Streams responses to the Streamlit chat UI  

```
Context + Query → Groq LLM → Streamed Response
```

---

### 🔹 4. Chat UI (Frontend)  
`app.py`  
- Streamlit-based chatbot  
- Maintains chat history  
- Displays suggested prompts  
- Real-time streaming of LLM answers  

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.10 |
| **Frameworks** | LangChain, Streamlit |
| **Vector DB** | Qdrant |
| **LLM Backend** | Groq API |
| **Embeddings** | SentenceTransformers |
| **Utilities** | dotenv, httpx, pypdf |

---

## Installation & Setup

### 1️. Clone the repository
```bash
git clone https://github.com/vijayvardhan6/hr-virtual-assistant.git
cd hr-virtual-assistant
```

### 2️. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
```

### 3️. Install dependencies
```bash
pip install -r requirements.txt
```

### 4️. Configure environment variables  
Copy the example file and update your keys:
```bash
cp .env.example .env
```

---

## Usage

### 1. Run the ingestion pipeline  
To load and embed all HR policy documents:
```bash
python ingestion_pipeline.py
```

### 2. Launch the chatbot interface  
Run the Streamlit app:
```bash
streamlit run app.py
```

### (Optional) Test retrieval performance  
```bash
python test_retrieve.py
```

---

## Project Structure

```
├── app.py                  # Streamlit chatbot UI
├── ingestion_pipeline.py   # Data ingestion & embedding
├── retrieval_pipeline.py   # Context retrieval logic
├── llm.py                  # Groq LLM interface
├── prompt_template.py      # System prompt definition
├── utils.py                # Utility functions
├── test_retrieve.py        # Retrieval pipeline tester
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧑‍💻 Example Query Flow

1. **User:** “What is the vacation policy?”  
2. **Retrieval:** Pipeline fetches related context from HR PDFs  
3. **LLM:** Groq model generates concise, context-grounded answer  
4. **UI:** Streamlit displays response in the chat interface  

---
