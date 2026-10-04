from typing import Any
import html
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = (5, 300)

USER_AVATAR = "👤"
ASSISTANT_AVATAR = "✨"

st.set_page_config(
    page_title="Enterprise Knowledge AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = {}

st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        visibility: hidden;
    }
    header { background: transparent !important; }

    .stApp { background: #121212; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 6rem; max-width: 1050px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1a1a1a; border-right: 1px solid #2d2d2d; }
    section[data-testid="stSidebar"] > div { padding: 1.2rem 0.9rem; }
    .brand { font-size: 19px; font-weight: 700; color: #ffffff; padding: 5px 10px 15px 10px; display: flex; align-items: center; }
    .brand-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 30px; height: 30px; margin-right: 10px; border-radius: 8px;
        background: linear-gradient(135deg, #6366f1, #a855f7); color: #ffffff; font-size: 16px; font-weight: 700;
    }
    .sidebar-heading {
        color: #9ca3af; font-size: 11px; font-weight: 700; letter-spacing: 1px;
        margin: 20px 10px 8px 10px; text-transform: uppercase;
    }
    .sidebar-description { color: #9ca3af; font-size: 12px; line-height: 1.5; padding: 0 10px; margin-bottom: 12px; }
    .sidebar-footer { color: #6b7280; font-size: 11px; padding: 10px; line-height: 1.5; }

    /* Buttons */
    .stButton > button {
        width: 100%; background: #262626; color: #f5f5f5; border: 1px solid #3f3f46;
        border-radius: 8px; min-height: 40px; font-size: 13px; font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { background: #323238; border-color: #6366f1; color: #ffffff; }

    /* File uploader */
    [data-testid="stFileUploader"] { background: #1e1e1e; border: 1px dashed #3f3f46; border-radius: 10px; padding: 8px; }

    /* Header */
    .top-header { display: flex; align-items: center; justify-content: space-between; padding: 0px 5px 20px 5px; border-bottom: 1px solid #27272a; margin-bottom: 20px; }
    .top-title { font-size: 20px; font-weight: 700; color: #ffffff; }
    .top-status { display: inline-block; margin-left: 10px; padding: 3px 10px; border-radius: 12px; background: #27272a; color: #a1a1aa; font-size: 11px; font-weight: 600; }

    /* Welcome */
    .welcome { text-align: center; padding-top: 12vh; padding-bottom: 12vh; }
    .welcome-logo {
        width: 64px; height: 64px; margin: 0 auto 20px auto; display: flex;
        align-items: center; justify-content: center; border-radius: 16px;
        background: linear-gradient(135deg, #6366f1, #a855f7); color: #ffffff; font-size: 30px; font-weight: 700;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
    }
    .welcome-title { font-size: 28px; font-weight: 700; color: #ffffff; margin-bottom: 8px; }
    .welcome-subtitle { color: #9ca3af; font-size: 14px; max-width: 500px; margin: 0 auto; }

    /* Chat Messages */
    [data-testid="stChatMessage"] { background: transparent; padding: 16px 0; border-bottom: 1px solid #1f1f23; }
    [data-testid="stChatMessage"] p { font-size: 15px; line-height: 1.6; color: #e4e4e7; }

    /* Sources */
    .sources-title { color: #a1a1aa; font-size: 12px; font-weight: 700; margin-top: 14px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    .source-card { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 10px 14px; margin: 6px 0; }
    .source-name { color: #f4f4f5; font-size: 13px; font-weight: 600; }
    .source-details { color: #71717a; font-size: 11px; margin-top: 4px; }

    /* Chat Input */
    [data-testid="stChatInput"] { background: #121212; }
    [data-testid="stChatInput"] > div { background: #18181b; border: 1px solid #27272a; border-radius: 12px; }
    [data-testid="stChatInput"] textarea { color: #f4f4f5 !important; background: transparent !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #71717a !important; }
    [data-testid="stChatInput"] > div:focus-within { border-color: #6366f1; box-shadow: 0 0 0 1px #6366f1; }

    .status-text { color: #71717a; font-size: 12px; padding: 4px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_error_detail(response: requests.Response, fallback: str) -> str:
    try:
        return response.json().get("detail", fallback)
    except Exception:
        return fallback


def upload_document(uploaded_file: Any) -> tuple[dict[str, Any] | None, str | None]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    try:
        res = requests.post(f"{API_URL}/upload", files=files, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        return None, "FastAPI server unreachable. Verify backend API process is running."
    except requests.exceptions.Timeout:
        return None, "Upload request timed out."
    except Exception as exc:
        return None, f"Unexpected upload failure: {exc}"

    if res.status_code == 200:
        return res.json(), None
    return None, extract_error_detail(res, "Document indexing failed.")


def ask_api(question: str) -> dict[str, Any]:
    try:
        res = requests.post(
            f"{API_URL}/ask", json={"question": question, "top_k": 3}, timeout=TIMEOUT
        )
    except requests.exceptions.ConnectionError:
        return {
            "content": "⚠️ Cannot connect to FastAPI server. Please start the backend service.",
            "sources": [],
        }
    except requests.exceptions.Timeout:
        return {"content": "⚠️ Inference query timed out.", "sources": []}
    except Exception as exc:
        return {"content": f"⚠️ Query execution error: {exc}", "sources": []}

    if res.status_code == 200:
        data = res.json()
        return {
            "content": data.get("answer", "No response content generated."),
            "sources": data.get("sources", []),
        }
    return {
        "content": f"⚠️ {extract_error_detail(res, 'API query failed.')}",
        "sources": [],
    }


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    st.markdown('<div class="sources-title">Retrieved Grounding Sources</div>', unsafe_allow_html=True)
    seen = set()
    for s in sources:
        name = str(s.get("source", "Unknown"))
        page = s.get("page")
        if (name, page) in seen:
            continue
        seen.add((name, page))

        score = s.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        page_str = f"Page {page}" if page is not None else "Page N/A"

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-name">📄 {html.escape(name)}</div>
                <div class="source-details">
                    {page_str} &nbsp;•&nbsp; Cosine Similarity: {score_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_message(message: dict[str, Any]) -> None:
    role = message["role"]
    avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if role == "assistant":
            render_sources(message.get("sources", []))


with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-icon">✦</span>Enterprise Knowledge AI</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-heading">Document Management</div>
        <div class="sidebar-description">
            Upload PDF, DOCX, or TXT enterprise documents to ingest into the FAISS vector database.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(
            f'<div class="status-text">📎 {len(uploaded_files)} file(s) selected</div>',
            unsafe_allow_html=True,
        )
        if st.button("Index Selected Documents", use_container_width=True):
            successful, failed = 0, 0
            progress_bar = st.progress(0)

            for idx, file in enumerate(uploaded_files):
                file_key = f"{file.name}:{file.size}"
                if file_key in st.session_state.indexed:
                    continue

                with st.spinner(f"Indexing {file.name}..."):
                    result, err = upload_document(file)

                if err:
                    failed += 1
                    st.error(f"{file.name}: {err}")
                else:
                    st.session_state.indexed[file_key] = result
                    successful += 1
                    st.success(f"Indexed {file.name}")

                progress_bar.progress((idx + 1) / len(uploaded_files))

            if successful:
                st.info(f"Successfully indexed {successful} document(s).")
            if failed:
                st.warning(f"Failed to index {failed} document(s).")

    st.markdown('<div class="sidebar-heading">Active Sessions</div>', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown('<div class="status-text">No active conversation.</div>', unsafe_allow_html=True)
    else:
        first_q = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"),
            "Active Chat",
        )
        display_q = first_q[:30] + "..." if len(first_q) > 30 else first_q
        st.markdown(f'<div class="status-text">💬 {html.escape(display_q)}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        """
        <div class="sidebar-footer">
            <strong>Enterprise Knowledge AI</strong><br>
            RAG • FAISS • SentenceTransformers • Llama 3.2
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="top-header">
        <div>
            <span class="top-title">Enterprise Knowledge AI</span>
            <span class="top-status">Local Vector RAG</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-logo">✦</div>
            <div class="welcome-title">Enterprise Intelligence Assistant</div>
            <div class="welcome-subtitle">Ask questions, query company policies, or search uploaded documents with verified citation grounding.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    render_message(msg)

prompt = st.chat_input("Ask Enterprise Knowledge AI...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Retrieving sources & synthesizing answer..."):
            response_data = ask_api(prompt)

    st.session_state.messages.append({"role": "assistant", **response_data})
    st.rerun()