import os
from dotenv import load_dotenv
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# Load env
#env_path = Path(__file__).resolve().parent / ".env"
#load_dotenv(env_path)
load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
API_KEY_2 = os.getenv("NVIDIA_API_KEY_2")



if not API_KEY:
    raise ValueError("API key missing")

# ---------------- GET TRANSCRIPT ----------------
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join([t.text for t in transcript])
        return text
    except TranscriptsDisabled:
        return None


# ---------------- BUILD VECTOR STORE ----------------
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2",
        api_key=API_KEY_2
    )

    db = FAISS.from_documents(chunks, embeddings)
    return db


# ---------------- ASK FUNCTION ----------------
def ask_video(db, question):
    retriever = db.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    final_prompt = prompt.format(context=context, question=question)

    llm = ChatNVIDIA(
        model="openai/gpt-oss-120b",
        api_key=API_KEY
    )

    response = llm.invoke(final_prompt)

    return response.content, docs

#transcript = get_transcript("Gfr50f6ZBvo")
#db = build_vector_store(transcript)
#question = "What is the video about?"
#print(ask_video(db, question))