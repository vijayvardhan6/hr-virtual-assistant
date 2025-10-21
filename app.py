import streamlit as st
import time
from retrieval_pipeline import retrieve_context
from llm import query_llm_with_history
from prompt_template import SYSTEM_PROMPT

# ----------------------------
# Timed performance tracker
# ----------------------------
def timed(label, store):
    t = time.time()
    def _end():
        store[label] = time.time() - t
    return _end

@st.cache_data(ttl=300, show_spinner=False)
def cached_retrieval(q: str):
    return retrieve_context(q)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="HR Virtual Assistant",
    layout="centered",
    page_icon="💬"
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 🧭 How to use:")
    st.markdown("""
    - Ask questions about **HR policies, benefits, or procedures**  
    - Get instant answers from the company knowledge base  
    - For personal matters, contact HR directly
    """)

# ----------------------------
# Styles
# ----------------------------
st.markdown("""
    <style>
        /* Remove the big default paddings and keep things centered */
        [data-testid="stAppViewContainer"] .main {
            padding-top: 1.25rem;      /* was ~3.5rem */
            padding-bottom: 0.75rem;   /* was large due to chat area */
        }
        /* Make the header transparent and slim */
        [data-testid="stHeader"] {
            background: transparent;
        }

        /* Constrain the main block width for nicer reading */
        .block-container {
            max-width: 960px;          /* adjust to taste: 880–1080px works well */
            padding-top: 0rem;         /* we already control above via .main */
            padding-bottom: 0rem;
        }

        /* Sidebar: tighten top spacing a bit */
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 0.75rem;
        }

        /* Typography + spacing for hero area */
        .main-title {
            text-align:center;
            font-size:36px;
            font-weight:700;
            color:#FF6600;
            margin: 0.25rem 0 0.25rem 0; /* tighter */
            line-height: 1.1;
        }
        .subtitle {
            text-align:center;
            color:#6b7280;             /* slightly softer gray */
            margin: 0 0 0.75rem 0;      /* tighter */
            font-size: 15.5px;
        }

        /* Suggestions: responsive grid, consistent gaps */
        .suggestion-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.6rem;
            margin: 0.75rem auto 0.25rem auto;
            max-width: 820px;
            margin-bottom: 1.5rem; 
        }
        .suggestion-btn {
            width: 100%;
            background:#f7f9fc;
            border:1px solid #e5e7eb;
            color:#374151;
            border-radius:10px;
            padding:0.6rem 0.9rem;
            font-size:0.95rem;
            cursor:pointer;
            transition: transform .06s ease, background .2s ease, border-color .2s ease;
        }
        .suggestion-btn:hover {
            background:#eef2ff;
            border-color:#93c5fd;
            color:#1e3a8a;
            transform: translateY(-1px);
        }

        /* Tighten spacing around chat messages a bit (optional) */
        [data-testid="stChatMessageContent"] p {
            margin-bottom: 0.35rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h2 class="main-title">Welcome to HR Virtual Assistant</h2>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about company policies, benefits, and procedures!</p>', unsafe_allow_html=True)

st.markdown("""
    <div class="suggestion-container">
        <button class="suggestion-btn">What is the vacation policy?</button>
        <button class="suggestion-btn">How many annual earned leaves?</button>
        <button class="suggestion-btn">What are the benefits?</button>
    </div>
""", unsafe_allow_html=True)



if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_query = st.chat_input("Ask about HR policies, benefits, procedures...")
timings = {}

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    end_retr = timed("retrieve", timings)
    context = cached_retrieval(user_query)
    end_retr()

    llm_messages = []
    
    # Add system message
    llm_messages.append(st.session_state.messages[0])
    
    # Add all previous conversation turns (user-assistant pairs)
    for msg in st.session_state.messages[1:-1]:
        llm_messages.append(msg)
    
    # Add current user query with context
    llm_messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nUser Query: {user_query}"
    })

    # Query the LLM and stream to the UI
    with st.chat_message("assistant"):
        end_llm = timed("llm_total", timings)
        streamed = query_llm_with_history(llm_messages, stream=True)
        answer_text = st.write_stream(streamed) or ""
        end_llm()

    # Persist the assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    