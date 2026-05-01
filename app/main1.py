import streamlit as st
import os

# Import both modules
from pdf import build_vector_store as build_pdf_db, ask_pdf
from yt_rag import get_transcript, build_vector_store as build_yt_db, ask_video

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Learning Platform", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
body { background-color: #0f172a; }

.user-msg {
    background: #3b82f6;
    padding: 10px;
    border-radius: 10px;
    text-align: right;
    margin: 5px 0;
    color: white;
}

.bot-msg {
    background: #1e293b;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    color: white;
}

.card {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "source_type" not in st.session_state:
    st.session_state.source_type = None

if "docs" not in st.session_state:
    st.session_state.docs = []

# ---------------- HEADER ----------------
st.title("🚀 AI Learning Platform")

# ---------------- SOURCE SELECTION ----------------
source = st.radio("Choose Source", ["PDF", "YouTube"])

# ---------------- PDF UPLOAD ----------------
if source == "PDF":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing PDF..."):
            st.session_state.db = build_pdf_db("temp.pdf")

        st.session_state.source_type = "pdf"
        st.success("PDF ready!")

# ---------------- YOUTUBE INPUT ----------------
elif source == "YouTube":
    yt_url = st.text_input("Enter YouTube URL")

    if st.button("Load Video"):
        if yt_url:
            # Extract video ID
            if "v=" in yt_url:
                video_id = yt_url.split("v=")[-1]
            else:
                video_id = yt_url

            with st.spinner("Fetching transcript..."):
                text = get_transcript(video_id)

            if not text:
                st.error("No transcript available")
            else:
                with st.spinner("Processing video..."):
                    st.session_state.db = build_yt_db(text)

                st.session_state.source_type = "youtube"
                st.success("Video ready!")

# ---------------- CHAT ----------------
if st.session_state.db:

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

    user_input = st.text_input("Ask your question...")

    if st.button("Send"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                if st.session_state.source_type == "pdf":
                    answer, docs = ask_pdf(st.session_state.db, user_input)
                else:
                    answer, docs = ask_video(st.session_state.db, user_input)

            st.session_state.messages.append({"role": "bot", "content": answer})
            st.session_state.docs = docs

            st.rerun()

    # ---------------- SOURCES ----------------
    with st.expander("📚 View Sources"):
        if st.session_state.docs:
            for doc in st.session_state.docs:
                st.markdown(
                    f"<div class='card'>{doc.page_content[:300]}...</div>",
                    unsafe_allow_html=True
                )
        else:
            st.write("No sources yet.")

else:
    st.info("Upload a PDF or load a YouTube video to start.")