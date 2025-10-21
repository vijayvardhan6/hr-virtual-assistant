import os
import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from functools import lru_cache

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))  
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

@lru_cache(maxsize=1)
def _llm():
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        http_client=httpx.Client(timeout=30.0, verify=False),  
    )

def query_llm(full_prompt: str, stream: bool = False):
    llm = _llm()
    if stream:
        for chunk in llm.stream(full_prompt):
            yield getattr(chunk, "content", str(chunk))
    else:
        resp = llm.invoke(full_prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

def query_llm_with_history(messages: list, stream: bool = False):
    llm = _llm()
    
    langchain_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
    
    if stream:
        for chunk in llm.stream(langchain_messages):
            yield getattr(chunk, "content", str(chunk))
    else:
        resp = llm.invoke(langchain_messages)
        return resp.content if hasattr(resp, "content") else str(resp)