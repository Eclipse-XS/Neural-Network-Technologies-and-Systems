"""Production Streamlit Interface for Local Engineering Knowledge Copilot."""
import asyncio
import os
import sys
import uuid
from pathlib import Path
import streamlit as st

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parents[4]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import httpx
from labs.lab07_final_project.src.knowledge_copilot.api.dependencies import (
    get_embeddings,
    get_ingestion_service,
    get_llm_client,
    get_query_service,
    get_redis_adapter,
    get_session_service,
    get_sqlite_repo,
)
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, RetrievalMode


st.set_page_config(
    page_title="Local Engineering Knowledge Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing Local Knowledge Copilot services...")
def get_services():
    sqlite_repo = get_sqlite_repo()
    redis_adapter = get_redis_adapter()
    embeddings = get_embeddings()
    llm_client = get_llm_client()
    ingestion_service = get_ingestion_service()
    session_service = get_session_service()
    query_service = get_query_service()
    
    # Initialize index and default corpus if Redis is online
    if redis_adapter.ping():
        redis_adapter.initialize_store(embeddings)
        # Load coursework if not yet indexed
        lab_root = Path(__file__).resolve().parents[3]
        knowledge_dir = lab_root / "assets/knowledge"
        manifest_file = knowledge_dir / "manifest.json"
        if manifest_file.exists():
            import json
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifest_filenames = {item["workspace_filename"] for item in manifest}
            for item in manifest:
                fpath = knowledge_dir / item["workspace_filename"]
                if fpath.exists():
                    try:
                        ingestion_service.ingest_file(
                            filename=item["workspace_filename"],
                            content=fpath.read_bytes(),
                            collection="coursework",
                            document_id=item["source_id"],
                        )
                    except Exception:
                        pass
        # Also ingest engineering documents from subdirectories (matching FastAPI lifespan)
        for fpath in knowledge_dir.glob("**/*"):
            if fpath.is_file() and fpath.suffix.lower() in {".md", ".txt", ".pdf"} and fpath.name != "manifest.json" and fpath.name not in locals().get("manifest_filenames", set()):
                doc_id = fpath.stem
                if ingestion_service.sqlite_repo.get_document(doc_id) is None:
                    try:
                        ingestion_service.ingest_file(
                            filename=fpath.name,
                            content=fpath.read_bytes(),
                            collection="engineering",
                            document_id=doc_id,
                        )
                    except Exception:
                        pass
    return {
        "sqlite": sqlite_repo,
        "redis": redis_adapter,
        "ingestion": ingestion_service,
        "session": session_service,
        "query": query_service,
        "llm": llm_client,
    }


services = get_services()
sqlite_repo = services["sqlite"]
redis_adapter = services["redis"]
ingestion_service = services["ingestion"]
session_service = services["session"]
query_service = services["query"]
llm_client = services["llm"]

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_{uuid.uuid4().hex[:12]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_mode" not in st.session_state:
    st.session_state.retrieval_mode = "hybrid"

# Load history from SQLite if messages empty
if not st.session_state.messages:
    db_session = session_service.get_session(st.session_state.session_id)
    if db_session:
        for t in db_session.turns:
            st.session_state.messages.append({
                "role": "user",
                "content": t.user_input,
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": t.canonical_answer,
                "grounding_status": t.grounding_status.value,
                "citations": t.citations,
                "invalid_citations": t.invalid_citations,
                "tools": [tool.to_dict() for tool in t.tool_executions],
                "evidence": [e.to_dict() for e in t.evidence],
            })

# ----------------- Sidebar ----------------- #
with st.sidebar:
    st.title("🧠 Knowledge Copilot")
    st.caption("Privacy-oriented Local Agentic RAG")
    
    st.subheader("System Status")
    redis_online = redis_adapter.ping()
    st.write(f"**Redis Vector Store:** {'🟢 Online' if redis_online else '🔴 Offline (Docker port 6380)'}")
    
    llm_online = False
    try:
        llm_online = asyncio.run(llm_client.is_available())
    except Exception:
        pass
    st.write(f"**Local LLM (LM Studio):** {'🟢 Available' if llm_online else '🟡 Standby / Port 1234'}")
    
    docs = ingestion_service.list_documents()
    indexed_count = sum(1 for d in docs if d.status.value == "INDEXED")
    st.write(f"**Indexed Documents:** {indexed_count} ({sum(d.chunk_count for d in docs)} chunks)")
    
    st.divider()
    st.subheader("Execution Backend")
    backend_mode = st.radio(
        "Service Backend",
        options=["FastAPI HTTP API (Port 8000)", "Standalone In-Process"],
        index=0,
        help="Select explicit execution path. HTTP API tests the networked FastAPI service, In-Process runs embedded.",
    )
    
    st.divider()
    st.subheader("Session Management")
    st.code(st.session_state.session_id, language=None)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = f"web_{uuid.uuid4().hex[:12]}"
            st.session_state.messages = []
            st.rerun()
            
    existing_sessions = session_service.list_sessions()
    session_options = [s.session_id for s in existing_sessions]
    if session_options:
        selected_session = st.selectbox(
            "Resume Session",
            options=[""] + session_options,
            index=0,
            help="Switch to an existing conversation history.",
        )
        if selected_session and selected_session != st.session_state.session_id:
            st.session_state.session_id = selected_session
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.subheader("Retrieval Settings")
    st.session_state.retrieval_mode = st.radio(
        "Retrieval Mode",
        options=["hybrid", "dense", "lexical"],
        index=0,
        help="Select hybrid (RRF), dense (multilingual-e5), or lexical (BM25).",
    )

# ----------------- Main Layout Tabs ----------------- #
tab_chat, tab_docs, tab_diagnostics = st.tabs(["💬 Chat & Copilot", "📚 Knowledge Management", "📊 Diagnostics & Telemetry"])

# TAB 1: Chat & Copilot
with tab_chat:
    st.header("Document-Grounded Q&A")
    st.caption("Ask questions about indexed engineering reports and technical specifications.")

    # Render Conversation Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                # Render Grounding Badge
                status_val = msg.get("grounding_status", "GROUNDED")
                badge_color = {
                    "GROUNDED": "green",
                    "PARTIALLY_SUPPORTED": "orange",
                    "INSUFFICIENT_EVIDENCE": "blue",
                    "VALIDATION_FAILED": "red",
                    "EXECUTION_FAILED": "red",
                }.get(status_val, "gray")
                st.caption(f":{badge_color}[● Grounding Status: {status_val}]")

                # Rich Evidence Viewer (DEFECT-04)
                evidence = msg.get("evidence", [])
                if evidence:
                    with st.expander(f"📑 Supporting Evidence ({len(evidence)} passages retrieved)"):
                        for idx, ev in enumerate(evidence, 1):
                            st.markdown(f"**[{idx}] {ev.get('source_id', 'doc')} — {ev.get('section', 'General')}** `(Citation: {ev.get('citation')})`")
                            st.info(ev.get("text", ""))
                            score_info = f"Method: `{ev.get('retrieval_method')}` | Score: `{ev.get('score_value')}` ({ev.get('score_type')}) | Rank: `#{ev.get('ranking_position')}`"
                            st.caption(score_info)
                            st.divider()

                # Tool Execution Details (DEFECT-03)
                tools = msg.get("tools", [])
                if tools:
                    with st.expander(f"🛠️ Tool Execution Trace ({len(tools)} calls)"):
                        for t in tools:
                            origin_str = t.get("origin", "MODEL_REQUESTED")
                            st.write(f"- Tool: **`{t.get('tool_name')}`** (`Origin: {origin_str}` | `{t.get('execution_time_ms')}ms`)")
                            st.json(t.get("arguments", {}))

    # User Input
    if user_query := st.chat_input("Ask a question about the uploaded technical documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, retrieving evidence, and verifying claims..."):
                try:
                    mode_str = st.session_state.retrieval_mode.lower()
                    api_url = os.getenv("COPILOT_API_URL", "http://127.0.0.1:8000")
                    chat_payload = {
                        "session_id": st.session_state.session_id,
                        "query": user_query,
                        "retrieval_mode": mode_str,
                    }

                    if backend_mode.startswith("FastAPI"):
                        try:
                            with httpx.Client(base_url=api_url, timeout=120.0) as http_client:
                                resp = http_client.post("/api/v1/chat", json=chat_payload)
                                if resp.status_code != 200:
                                    err_detail = resp.text
                                    st.error(
                                        f"FastAPI HTTP Error ({resp.status_code}): {err_detail}\n\n"
                                        "Ensure the FastAPI service is running or switch to 'Standalone In-Process' backend."
                                    )
                                    st.stop()
                                http_resp_data = resp.json()
                        except Exception as conn_err:
                            st.error(
                                f"Could not connect to FastAPI at `{api_url}`: {conn_err}\n\n"
                                "Start the service via:\n"
                                "```bash\nuvicorn labs.lab07_final_project.src.knowledge_copilot.api.app:app --host 127.0.0.1 --port 8000\n```\n"
                                "Or select 'Standalone In-Process' in the sidebar."
                            )
                            st.stop()

                        ans_text = http_resp_data["answer"]
                        status_val = http_resp_data["grounding_status"]
                        evidence_list = http_resp_data.get("evidence", [])
                        tools_list = http_resp_data.get("tools_used", [])
                        citations = http_resp_data.get("citations", [])
                        invalid_citations = http_resp_data.get("invalid_citations", [])
                    else:
                        mode_enum = RetrievalMode(st.session_state.retrieval_mode)
                        turn = asyncio.run(
                            query_service.execute_query(
                                query=user_query,
                                session_id=st.session_state.session_id,
                                retrieval_mode=mode_enum,
                            )
                        )
                        ans_text = turn.canonical_answer
                        status_val = turn.grounding_status.value
                        evidence_list = [e.to_dict() for e in turn.evidence]
                        tools_list = [t.to_dict() for t in turn.tool_executions]
                        citations = turn.citations
                        invalid_citations = turn.invalid_citations
                    
                    st.markdown(ans_text)
                    
                    # Display Status Badge
                    badge_color = {
                        "GROUNDED": "green",
                        "PARTIALLY_SUPPORTED": "orange",
                        "INSUFFICIENT_EVIDENCE": "blue",
                        "VALIDATION_FAILED": "red",
                        "EXECUTION_FAILED": "red",
                    }.get(status_val, "gray")
                    st.caption(f":{badge_color}[● Grounding Status: {status_val}]")

                    # Evidence Viewer
                    if evidence_list:
                        with st.expander(f"📑 Supporting Evidence ({len(evidence_list)} passages retrieved)"):
                            for idx, ev in enumerate(evidence_list, 1):
                                st.markdown(f"**[{idx}] {ev.get('source_id', '')} — {ev.get('section', '')}** `(Citation: {ev.get('citation', '')})`")
                                st.info(ev.get("text", ""))
                                score_info = f"Method: `{ev.get('retrieval_method', '')}` | Score: `{round(ev.get('score_value', 0.0), 4)}` ({ev.get('score_type', '')}) | Rank: `#{ev.get('ranking_position', '')}`"
                                st.caption(score_info)
                                st.divider()

                    # Tool Traces
                    if tools_list:
                        with st.expander(f"🛠️ Tool Execution Trace ({len(tools_list)} calls)"):
                            for t in tools_list:
                                st.write(f"- Tool: **`{t.get('tool_name')}`** (`Origin: {t.get('origin')}` | `{round(float(t.get('execution_time_ms', 0.0)), 1)}ms`)")
                                st.json(t.get("arguments", {}))

                    # Append to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans_text,
                        "grounding_status": status_val,
                        "citations": citations,
                        "invalid_citations": invalid_citations,
                        "tools": tools_list,
                        "evidence": evidence_list,
                    })

                except Exception as exc:
                    err_msg = f"Backend Execution Error: {type(exc).__name__}: {exc}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "grounding_status": "EXECUTION_FAILED",
                        "citations": [],
                        "invalid_citations": [],
                        "tools": [],
                        "evidence": [],
                    })


# TAB 2: Knowledge Management
with tab_docs:
    st.header("Document Collection & Ingestion Lifecycle")
    st.caption("Upload technical documents (.md, .txt, text-based .pdf). Scanned PDFs require an OCR layer.")

    uploaded_files = st.file_uploader(
        "Upload Technical Documentation",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            if st.button(f"Index '{f.name}'", key=f"btn_{f.name}"):
                with st.spinner(f"Ingesting and indexing '{f.name}'..."):
                    try:
                        content_bytes = f.read()
                        doc = ingestion_service.ingest_file(
                            filename=f.name,
                            content=content_bytes,
                            collection="user_uploads",
                        )
                        st.success(f"Successfully indexed '{doc.filename}' ({doc.chunk_count} chunks generated).")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingestion failed for '{f.name}': {type(exc).__name__}: {exc}")

    st.subheader("Storage Consistency & Maintenance")
    if st.button("Reconcile Storage (Repair Inconsistencies)", type="secondary"):
        with st.spinner("Checking SQLite and Redis alignment..."):
            rec_report = ingestion_service.reconcile_storage()
            st.json(rec_report)

    st.subheader("Current Document Index")
    all_docs = ingestion_service.list_documents()
    
    if all_docs:
        for d in all_docs:
            col_info, col_actions = st.columns([4, 1])
            with col_info:
                badge = "🟢 INDEXED" if d.status.value == "INDEXED" else "🔴 FAILED"
                st.markdown(f"**{d.document_id}**: `{d.filename}` ({d.size_bytes} bytes, {d.chunk_count} chunks) — {badge}")
                if d.error_message:
                    st.caption(f":red[Error: {d.error_message}]")
            with col_actions:
                if st.button("Delete", key=f"del_{d.document_id}", type="secondary"):
                    try:
                        ingestion_service.delete_document(d.document_id)
                        st.toast(f"Deleted {d.document_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete {d.document_id}: {e}")
            st.divider()
    else:
        st.info("No documents indexed yet. Upload documents or start Redis to load default coursework.")


# TAB 3: Diagnostics & Telemetry
with tab_diagnostics:
    st.header("Service Health & Observability")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Redis Status", "Connected" if redis_online else "Disconnected", "127.0.0.1:6380")
    with col_d2:
        st.metric("Indexed Documents", str(len(all_docs)))
    with col_d3:
        total_chunks = sum(d.chunk_count for d in all_docs)
        st.metric("Total Active Chunks", str(total_chunks))

    st.subheader("Prometheus Metrics Endpoint")
    st.write("Live Prometheus metrics are published at `/metrics` via FastAPI.")
    st.code(
        """
        # Minimum metrics tracked:
        rag_requests_total
        rag_request_duration_seconds
        rag_retrieval_duration_seconds
        rag_retrieval_empty_total
        rag_tool_calls_total{origin="MODEL_REQUESTED|RECOVERY_EXECUTED"}
        rag_citation_validation_failures_total
        rag_indexed_documents
        rag_indexed_chunks
        rag_ingestion_failures_total
        """,
        language="text",
    )
