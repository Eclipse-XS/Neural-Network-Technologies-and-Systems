"""Legacy Compatibility Streamlit UI for Local Engineering Knowledge Copilot.

Canonical multi-tab Streamlit entrypoint:
  streamlit run labs/lab07_final_project/src/knowledge_copilot/ui/streamlit_app.py --server.port 8501
"""
import asyncio
import os
import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.agent import connect, create_agent, create_session, execute
from src.config import Settings
from src.knowledge_base import KnowledgeBase
from src.knowledge_copilot.api.dependencies import get_ingestion_service
from src.knowledge_copilot.rag.extractors import extract_text_from_bytes, validate_file_content

st.set_page_config(page_title="Coursework Knowledge Assistant", layout="wide")
st.title("Coursework Knowledge Assistant")
st.caption("Local Engineering Knowledge Copilot. Grounded retrieval, safe calculations, and isolated memory.")


@st.cache_resource(show_spinner="Validating corpus and opening the Redis index…")
def knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase(Settings())
    kb.initialize()
    return kb


async def ask(question: str, session_id: str) -> dict:
    kb = knowledge_base()
    client, settings = await connect(Settings())
    try:
        return await execute(
            create_agent(client, settings, kb),
            settings,
            question,
            session=create_session(settings, session_id),
        )
    finally:
        await client.close()


if "session_id" not in st.session_state:
    st.session_state.session_id = "web_" + __import__("uuid").uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
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
        kb_instance = knowledge_base()
        for source in kb_instance.sources():
            st.write(f"📄 **{source['source_id']}**: {source.get('title', source.get('workspace_filename'))}")
    except Exception as exc:
        st.error(f"Backend unavailable: {type(exc).__name__}: {exc}")

    st.divider()
    with st.expander("📤 Ingest New Document"):
        st.caption("Supports .md, .txt, text-based .pdf")
        uploaded_file = st.file_uploader("Upload Document", type=["md", "txt", "pdf"])
        if uploaded_file and st.button("Index Document", use_container_width=True):
            try:
                data = uploaded_file.read()
                clean_name, sha256_hash, size_bytes = validate_file_content(data, uploaded_file.name)
                extracted_text = extract_text_from_bytes(data, clean_name)
                doc = get_ingestion_service().ingest_file(
                    filename=clean_name,
                    content=data,
                    collection="user_uploads",
                )
                st.success(
                    f"Validated and indexed '{doc.filename}' ({size_bytes} bytes, {doc.chunk_count} chunks)."
                )
            except Exception as e:
                st.error(f"Ingestion failed: {type(e).__name__}: {e}")

# Render Conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display Grounding Badge
        if message.get("grounding_status"):
            st_val = message["grounding_status"]
            b_color = "green" if st_val == "GROUNDED" else "orange" if st_val == "PARTIALLY_SUPPORTED" else "blue"
            st.caption(f":{b_color}[● Grounding Status: {st_val}]")
            
        # Rich Evidence Viewer (DEFECT-04)
        evidence = message.get("evidence", [])
        if evidence:
            with st.expander(f"📑 Supporting Evidence ({len(evidence)} passages)"):
                for idx, ev in enumerate(evidence, 1):
                    st.markdown(f"**[{idx}] {ev.get('source_id', 'doc')} — {ev.get('section', 'General')}** `(Citation: {ev.get('citation')})`")
                    st.info(ev.get("text", ""))
                    st.caption(f"Score: `{round(ev.get('cosine_distance', 0.0), 4)}` | Evidence Quality: `{ev.get('evidence_quality')}`")
                    st.divider()

        if message.get("details"):
            with st.expander("🛠️ Tool Trace Details"):
                st.json(message["details"])

# Chat Input
if question := st.chat_input("Ask about Labs 3–6 or uploaded documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Running local agent and verifying grounding…"):
            try:
                record = asyncio.run(ask(question, st.session_state.session_id))
                answer = record["final_answer"] if record["status"] == "SUCCESS" else record["error"]
                
                details = {
                    k: record[k]
                    for k in (
                        "tools_used",
                        "sources_retrieved",
                        "citation",
                        "invalid_citations",
                        "runtime_seconds",
                        "grounding_status",
                    )
                    if k in record
                }
                grounding_status = record.get("grounding_status", "GROUNDED")
                evidence_list = record.get("evidence", [])

                st.markdown(answer)
                
                b_color = "green" if grounding_status == "GROUNDED" else "orange" if grounding_status == "PARTIALLY_SUPPORTED" else "blue"
                st.caption(f":{b_color}[● Grounding Status: {grounding_status}]")
                
                if evidence_list:
                    with st.expander(f"📑 Supporting Evidence ({len(evidence_list)} passages)"):
                        for idx, ev in enumerate(evidence_list, 1):
                            st.markdown(f"**[{idx}] {ev.get('source_id', 'doc')} — {ev.get('section', 'General')}** `(Citation: {ev.get('citation')})`")
                            st.info(ev.get("text", ""))
                            st.caption(f"Score: `{round(ev.get('cosine_distance', 0.0), 4)}` | Quality: `{ev.get('evidence_quality')}`")
                            st.divider()

                with st.expander("🛠️ Tool Trace Details"):
                    st.json(details)

            except Exception as exc:
                answer = f"Backend unavailable: {type(exc).__name__}: {exc}"
                details = None
                evidence_list = []
                grounding_status = "EXECUTION_FAILED"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "details": details,
        "evidence": evidence_list,
        "grounding_status": grounding_status,
    })
