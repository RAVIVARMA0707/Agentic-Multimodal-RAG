import streamlit as st
import requests

# ---------------------------
# API CONFIG
# ---------------------------
CHAT_API_URL = "http://127.0.0.1:8000/api/v1/query"
UPLOAD_API_URL = "http://127.0.0.1:8000/api/v1/upload-pdf"

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
with st.sidebar:
    st.title("🧠 AI Assistant")
    page = st.radio("Navigation", ["Chat", "Admin"])
    st.divider()

# =========================================================
# 💬 CHAT PAGE
# =========================================================
if page == "Chat":

    st.title("AI Document Question Answering")

    # ---------------------------
    # SESSION STATE INIT
    # ---------------------------
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"Chat 1": []}
        st.session_state.current_chat = "Chat 1"

    # ---------------------------
    # CHAT LIST (SIDEBAR)
    # ---------------------------
    with st.sidebar:
        st.markdown("### 💬 Chats")

        if st.button("New Chat"):
            new_chat = f"Chat {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[new_chat] = []
            st.session_state.current_chat = new_chat

        for chat in st.session_state.chat_sessions:
            if st.button(chat):
                st.session_state.current_chat = chat

    # ---------------------------
    # LOAD CURRENT CHAT
    # ---------------------------
    current_chat = st.session_state.current_chat
    messages = st.session_state.chat_sessions[current_chat]

    st.markdown(f"### 💬 {current_chat}")
    st.divider()

    # ---------------------------
    # DISPLAY MESSAGES (PERSISTENT)
    # ---------------------------
    if not messages:
        st.info("👋 Upload documents in **Admin** tab and start asking questions.")

    for msg in messages:
        with st.chat_message(msg["role"]):

            # USER MESSAGE
            if msg["role"] == "user":
                st.markdown(msg["content"])

            # ASSISTANT MESSAGE
            else:
                # Answer text
                st.markdown(msg.get("answer", ""))

                # Image (logo / picture) if present ✅
                if msg.get("image_path"):
                    st.image(
                        msg["image_path"],
                        use_column_width=True,
                        caption="Retrieved image from document"
                    )

                # Metadata (independent rendering so it never vanishes)
                meta_parts = []
                if msg.get("document_name"):
                    meta_parts.append(f"📄 {msg['document_name']}")
                if msg.get("page_no"):
                    meta_parts.append(f"📘 Page : {msg['page_no']}")

                if meta_parts:
                    st.markdown(" | ".join(meta_parts))

    # ---------------------------
    # USER INPUT
    # ---------------------------
    user_input = st.chat_input("Ask a question about the uploaded documents...")

    if user_input:
        # Save user message
        messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # ---------------------------
        # API CALL
        # ---------------------------
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                try:
                    response = requests.post(
                        CHAT_API_URL,
                        json={
                            "query": user_input,
                            "k": 5
                        },
                        timeout=120
                    )

                    result = response.json()

                    answer = result.get("answer", "No answer found.")
                    image_path = result.get("image_path")
                    doc_name = result.get("document_name")
                    page_no = result.get("page_no")

                    # Render immediately
                    st.markdown(answer)

                    if image_path:
                        st.image(
                            image_path,
                            use_column_width=True,
                            caption="Retrieved image from document"
                        )

                    meta_parts = []
                    if doc_name:
                        meta_parts.append(f"📄 {doc_name}")
                    if page_no:
                        meta_parts.append(f"📘 Page : {page_no}")

                    if meta_parts:
                        st.markdown(" | ".join(meta_parts))

                    # ✅ Persist full assistant message (text + image + metadata)
                    messages.append({
                        "role": "assistant",
                        "answer": answer,
                        "image_path": image_path,
                        "document_name": doc_name,
                        "page_no": page_no
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)

                    messages.append({
                        "role": "assistant",
                        "answer": error_msg,
                        "image_path": None,
                        "document_name": None,
                        "page_no": None
                    })

# =========================================================
# 🛠️ ADMIN PAGE
# =========================================================
elif page == "Admin":

    st.title("Admin Dashboard")

    password = st.text_input("Enter Admin Password", type="password")

    if password != "admin123":
        st.warning("🔒 Admin access only")
        st.stop()

    st.success("✅ Access Granted")

    st.markdown("### 📄 Upload Knowledge Base PDFs")
    st.markdown(
        "Uploading new PDFs will **automatically delete all previously "
        "ingested documents, embeddings, and images** before ingestion."
    )

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Ingest Documents"):
        with st.spinner("Resetting knowledge base and ingesting PDFs..."):
            try:
                files = [
                    ("files", (f.name, f, "application/pdf"))
                    for f in uploaded_files
                ]

                response = requests.post(
                    UPLOAD_API_URL,
                    files=files,
                    timeout=300
                )

                if response.status_code == 200:
                    st.success(
                        f"✅ {len(uploaded_files)} PDFs ingested successfully.\n\n"
                        "🧹 Previous data cleared automatically."
                    )
                else:
                    st.error(
                        f"❌ Upload failed (status {response.status_code})"
                    )

            except Exception as e:
                st.error(f"❌ Error during upload: {str(e)}")
