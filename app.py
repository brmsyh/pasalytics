"""Streamlit chat front-end for Pasalytics.

Run locally:  streamlit run app.py
On Streamlit Cloud, add GEMINI_API_KEY under App settings > Secrets."""
import os

import streamlit as st

try:  # Streamlit Cloud secrets (GEMINI_API_KEY, GEMINI_CHAT_MODEL...) -> environment, before config is imported
    for key, value in st.secrets.items():
        if key.startswith("GEMINI_"):
            os.environ[key] = str(value)
except Exception:
    pass

from src import config  # noqa: E402
from src.generator import GeminiGenerator, RAGPipeline  # noqa: E402
from src.retriever import build_retriever  # noqa: E402

st.set_page_config(page_title="Pasalytics", page_icon="⚖️", layout="centered")

EXAMPLES = [
    "How long do we have to respond to a provisional refusal?",
    "What is the hourly rate of a senior associate?",
    "When does the PCT national phase deadline fall for Indonesia?",
    "What happens on day 90 of an unpaid invoice?",
    "Can I put client documents into ChatGPT?",
]


@st.cache_resource(show_spinner="Loading index...")
def get_pipeline(mode: str):
    retriever = build_retriever(mode)
    generator = GeminiGenerator() if config.GEMINI_API_KEY else None
    return retriever, generator


with st.sidebar:
    st.header("Settings")
    mode = st.radio("Retrieval mode", ["hybrid", "dense", "bm25"], index=0,
                    help="Hybrid = BM25 + Gemini embeddings fused with Reciprocal Rank Fusion.")
    k = st.slider("Passages retrieved (top-k)", 3, 10, config.TOP_K)
    st.divider()
    st.caption("Knowledge base: 20 internal SOPs and policies of a fictional IP firm "
               "(synthetic) plus public Indonesian IP regulations.")
    st.caption("Answers describe firm procedures and are not legal advice.")

retriever, generator = get_pipeline(mode)
if retriever.name != mode:
    st.sidebar.warning(f"Running in {retriever.name.upper()} mode (no embeddings or API key found).")

st.title("⚖️ Pasalytics")
st.caption("Your IP firm knowledge assistant: procedures, deadlines, fees and regulations, with sources.")
st.write("Ask about filing procedures, deadlines, fees and internal policies. "
         "Every answer cites the passages it is based on.")

if "messages" not in st.session_state:
    st.session_state.messages = []

cols = st.columns(len(EXAMPLES))
clicked = None
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        clicked = ex

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question...") or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if generator is None:
            hits = retriever.search(question, k)
            reply = "**No API key configured - showing the top retrieved passages only.**"
            st.markdown(reply)
        else:
            with st.spinner("Searching and drafting..."):
                answer = RAGPipeline(retriever, generator, k).ask(question)
            hits, reply = answer.hits, answer.text
            st.markdown(reply)
        with st.expander("Sources"):
            for i, h in enumerate(hits, start=1):
                c = h.chunk
                page = f" · page {c['page']}" if c.get("page") else ""
                st.markdown(f"**[{i}] {c['doc_id']}** — {c['section']}{page}  \n"
                            f"<small>{c['text'][:400]}...</small>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": reply})
