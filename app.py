from src.retriever import Retriever
from src.loader import load_pdf, load_txt
from src.generator import answer_question
from src.extractor import analyze_contract
from src.chunker import chunk_pages
from src.config import GEMINI_API_KEY
import hashlib
import io
import os

import streamlit as st

st.set_page_config(page_title="Contract Assistant",
                   page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    html { font-size: 19px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# On Streamlit Cloud the API key comes from secrets; locally it comes from .env
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass  # no secrets file when running locally, which is fine


if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file (or to Streamlit secrets when deployed).")
    st.stop()


# above this, first-time indexing takes several minutes on the free API tier
CHUNK_WARN = 150
SUGGESTIONS = {
    "Termination": "How can this agreement be terminated?",
    "Payment": "What are the payment or compensation terms?",
    "Confidentiality": "What are the confidentiality obligations?",
    "Governing law": "Which law governs this agreement?",
}
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def esc(text: str) -> str:
    """Stop Streamlit from reading $...$ as math (contracts are full of dollar amounts)."""
    return text.replace("$", "\\$")


@st.cache_data(show_spinner=False, max_entries=5)
def read_contract(name: str, data: bytes):
    if name.lower().endswith(".txt"):
        pages = load_txt(io.BytesIO(data))
    else:
        pages = load_pdf(io.BytesIO(data))
    return pages, len(chunk_pages(pages))


@st.cache_resource(show_spinner=False, max_entries=3)
def build_retriever(key: str, _pages):
    return Retriever(_pages)


def render_sources(sources):
    if not sources:
        return
    st.caption("Sources used")
    for s in sources:
        with st.expander(f"{s['section']} — p.{s['page']} (similarity {s['score']})"):
            st.text(s["text"])


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(esc(msg["content"]))
        render_sources(msg.get("sources"))


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📄 Contract Assistant")
uploaded = st.sidebar.file_uploader("Upload a contract", type=["pdf", "txt"])
st.sidebar.caption("Informational only. Not legal advice.")

if uploaded is None:
    st.title("Legal Contract Analysis and Retrieval Assistant")
    st.info("Upload a contract (PDF or text) in the sidebar to get started.")
    st.stop()

data = uploaded.getvalue()
key = hashlib.sha256(data).hexdigest()[:16]

try:
    pages, n_chunks = read_contract(uploaded.name, data)
except Exception as e:
    st.error(f"Could not read this file: {e}")
    st.stop()

# Reset the chat and analysis when a different contract is uploaded
if st.session_state.get("doc_key") != key:
    st.session_state.update(doc_key=key, messages=[], analysis=None)

st.sidebar.caption(
    f"{uploaded.name}: {len(pages)} pages, {n_chunks} sections indexed")
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

if n_chunks > CHUNK_WARN and not st.session_state.get(f"ok_{key}"):
    st.warning(
        f"This contract has {n_chunks} sections. On the free Gemini API tier, indexing it for the "
        f"first time takes roughly {n_chunks // 90 + 1} minutes (it is saved afterwards)."
    )
    if st.button("Index it anyway"):
        st.session_state[f"ok_{key}"] = True
        st.rerun()
    st.stop()

with st.spinner("Indexing the contract (the first time can take a minute)..."):
    try:
        retriever = build_retriever(key, pages)
    except Exception as e:
        st.error(f"Could not index the contract: {e}")
        st.stop()

# ---------------------------------------------------------------- main area
st.title("Legal Contract Analysis and Retrieval Assistant")
tab_chat, tab_analysis = st.tabs(["💬 Ask questions", "📋 Contract analysis"])

with tab_chat:
    st.caption("Try one:")
    cols = st.columns(len(SUGGESTIONS))
    for col, (label, q) in zip(cols, SUGGESTIONS.items()):
        if col.button(label, key=f"sugg_{label}"):
            st.session_state.pending_q = q

    for msg in st.session_state.messages:
        render_message(msg)

    question = st.chat_input("Ask a question about this contract")
    if not question:
        question = st.session_state.pop("pending_q", None)

    if question:
        user_msg = {"role": "user", "content": question}
        st.session_state.messages.append(user_msg)
        render_message(user_msg)

        with st.chat_message("assistant"):
            result = None
            with st.spinner("Searching the contract..."):
                try:
                    result = answer_question(
                        question, retriever.retrieve(question))
                except Exception as e:
                    st.error(f"Sorry, something went wrong: {e}")
            if result:
                st.markdown(esc(result["answer"]))
                render_sources(result["sources"])
                st.session_state.messages.append(
                    {"role": "assistant",
                        "content": result["answer"], "sources": result["sources"]}
                )
            else:
                st.session_state.messages.pop()   # drop the unanswered question

with tab_analysis:
    st.write(
        "Extracts obligations, termination clauses and potential risks from the clauses "
        "most relevant to each topic."
    )
    if st.button("Analyze contract", type="primary"):
        with st.spinner("Analyzing the contract..."):
            try:
                st.session_state.analysis = analyze_contract(retriever)
            except Exception as e:
                st.error(f"Sorry, the analysis failed: {e}")

    analysis = st.session_state.get("analysis")
    if analysis:
        st.subheader("Obligations")
        if analysis.obligations:
            st.dataframe(
                [
                    {
                        "Party": o.party,
                        "Obligation": o.obligation,
                        "Deadline / condition": o.deadline_or_condition,
                        "Section": o.source_section,
                    }
                    for o in analysis.obligations
                ],
                hide_index=True,
            )
        else:
            st.caption("None found in the retrieved clauses.")

        st.subheader("Termination clauses")
        if analysis.termination_clauses:
            st.dataframe(
                [
                    {
                        "Trigger": t.trigger,
                        "Notice period": t.notice_period,
                        "Consequences": t.consequences,
                        "Section": t.source_section,
                    }
                    for t in analysis.termination_clauses
                ],
                hide_index=True,
            )
        else:
            st.caption("None found in the retrieved clauses.")

        st.subheader("Potential risks")
        if analysis.risks:
            for r in sorted(analysis.risks, key=lambda r: SEVERITY_ORDER[r.severity]):
                st.markdown(
                    f"{SEVERITY_ICON[r.severity]} **{r.severity.upper()}**: {esc(r.description)}")
                st.caption(esc(f"{r.reason} (Section: {r.source_section})"))
        else:
            st.caption("None found in the retrieved clauses.")

        st.download_button(
            "Download analysis (JSON)",
            analysis.model_dump_json(indent=2),
            file_name="contract_analysis.json",
            mime="application/json",
        )
        st.caption(
            "Risks are an automated first-pass opinion based on the retrieved clauses. Not legal advice.")
