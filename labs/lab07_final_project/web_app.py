"""Minimal Streamlit UI over the same KnowledgeBase and agent services as the CLI."""
import asyncio
import streamlit as st

from src.agent import connect, create_agent, create_session, execute
from src.config import Settings
from src.knowledge_base import KnowledgeBase


st.set_page_config(page_title="Coursework Knowledge Assistant", layout="wide")
st.title("Coursework Knowledge Assistant")
st.caption("Local RAG + tool-calling agent. Retrieved passages are evidence candidates, not hidden reasoning.")


@st.cache_resource(show_spinner="Validating corpus and opening the Redis index…")
def knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase(Settings())
    kb.initialize()
    return kb


async def ask(question: str, session_id: str) -> dict:
    kb = knowledge_base()
    client, settings = await connect(Settings())
    try:
        return await execute(create_agent(client, settings, kb), settings, question,
                             session=create_session(settings, session_id))
    finally:
        await client.close()


if "session_id" not in st.session_state:
    st.session_state.session_id = "web_" + __import__("uuid").uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.session_id)
    if st.button("New session", use_container_width=True):
        st.session_state.session_id = "web_" + __import__("uuid").uuid4().hex[:12]
        st.session_state.messages = []
        st.rerun()
    resume = st.text_input("Resume session ID")
    if st.button("Resume", use_container_width=True, disabled=not resume.strip()):
        st.session_state.session_id = resume.strip()
        st.session_state.messages = []
        st.rerun()
    st.subheader("Indexed documents")
    try:
        for source in knowledge_base().sources():
            st.write(f"{source['source_id']}: {source['title']}")
    except Exception as exc:
        st.error(f"Backend unavailable: {type(exc).__name__}: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            with st.expander("Tool trace and evidence"):
                st.json(message["details"])

if question := st.chat_input("Ask about Labs 3–6"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Running the local agent…"):
            try:
                record = asyncio.run(ask(question, st.session_state.session_id))
                answer = record["final_answer"] if record["status"] == "SUCCESS" else record["error"]
                details = {k: record[k] for k in ("tools_used", "sources_retrieved", "citation",
                                                   "invalid_citations", "runtime_seconds")}
                st.markdown(answer)
                with st.expander("Tool trace and evidence"):
                    st.json(details)
            except Exception as exc:
                answer = f"Backend unavailable: {type(exc).__name__}: {exc}"
                details = None
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
