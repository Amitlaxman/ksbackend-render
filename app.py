from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import os

from pymongo import MongoClient
import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["legal_db"]

# Helper to load Mongo docs
def load_mongo_documents(db, collections):
    docs = []
    for coll_name in collections:
        coll = db[coll_name]
        for doc in coll.find():
            text_en = doc.get("content_en", "")
            text_hi = doc.get("content_hi", "")
            combined = f"EN: {text_en}\n\nHI: {text_hi}"
            docs.append(Document(
                page_content=combined,
                metadata={"source": doc.get("source", coll_name), "state": doc.get("state", ""), "collection": coll_name}
            ))
    return docs

collections = [
    "Building_and_Other_Construction_Workers_Act_1996_Notification",
    "Building_and_Other_Construction_Workers_Act_1996_Adhisuchna",
    "Contract_Labour_Act_1970",
    "Contract_Labour_Act_1970_section6.2",
    "Contract_Labour_Act_1970_section_6.1",
    "Contract_Labour_MP_Rules_1973",
    "MP_Govt_Info",
    "MP_Govt_Notifications",
    "MP_Shops_and_Establishments_Act_1958_Night_Work_Rules",
    "MP_Shops_and_Establishments_Rules_1959_Notifications",
    "Payment_of_Gratuity_Act_1972_Notifications",
    "Shops_&_Establishments_Act_1958",
    "Shops_&_Establishments_Act_1958_Notification",
    "Shops_&_Establishments_Act_195_district"
] # keep your list


mongo_docs = load_mongo_documents(db, collections)

# Load CSV
DATA_FILE = "data.csv"  # place in project root
df = pd.read_csv(DATA_FILE)
csv_docs = [Document(page_content=" ".join([f"{col}: {row[col]}" for col in df.columns])) for _, row in df.iterrows()]

# Embeddings + retrievers
emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
csv_vectorstore = FAISS.from_documents(csv_docs, emb_model)
mongo_vectorstore = FAISS.from_documents(mongo_docs, emb_model)
csv_retriever = csv_vectorstore.as_retriever(search_kwargs={"k": 3})
mongo_retriever = mongo_vectorstore.as_retriever(search_kwargs={"k": 3})
combined_retriever = EnsembleRetriever(retrievers=[csv_retriever, mongo_retriever], weights=[0.4, 0.6])

# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

system_prompt = """
You are a legal assistant chatbot for workers in Madhya Pradesh.
            Only answer based on the context provided below.
            If the answer is not found, clearly say "I don't know based on the law.
Your job is to explain government schemes, worker rights, and benefits.

Instructions:
- Explain the answer in simple words.
- Try to cite your sources if you have any.
- Add an example or step-by-step explanation if possible.
- If user writes in Hindi, reply in Hindi. If in English, reply in English.
- Only answer questions that you know the answer surely to.
- Do not site incorrect sources.
- ONLY USE INFORMATION FROM MADHYA PRADESH GOVT.
- https://labour.mponline.gov.in/ this is a source that can be cited.
- https://drive.google.com/drive/folders/1F2sQADtqSS9ySo_44-kW_7UxzE3XZmxu this is another source.

-If the answer is not in the provided context, reply only with:
"I’m not sure, but I’ll pass this to an admin."
-Do NOT try to guess or invent.
"""

qa = RetrievalQA.from_chain_type(llm=llm, retriever=combined_retriever, chain_type="stuff")

def detect_language(text: str) -> str:
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    return "en"

# FastAPI
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: Query):
    try:
        lang = detect_language(query.query)
        if lang == "hi":
            prompt = f"{system_prompt}\n\nप्रश्न: {query.query}\nउत्तर हिंदी में दीजिए।"
        else:
            prompt = f"{system_prompt}\n\nQuestion: {query.query}\nAnswer in English."
        answer = qa.invoke(prompt)["result"]
        return {"response": answer}
    except Exception as e:
        return {"response": f"⚠️ Error: {str(e)}"}
