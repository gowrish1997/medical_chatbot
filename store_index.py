from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
pc=Pinecone(api_key=PINECONE_API_KEY)
index_name="medical-chatbot"

loaded_pdf_data=load_pdf_files('data')
filtered_docs=filter_to_minimal_docs(loaded_pdf_data)
texts_chunk=text_split(filtered_docs)
embedding=download_embeddings()

if not pc.has_index(index_name):
   pc.create_index(
      name=index_name,
      dimension=384,
      metric='cosine',
      spec=ServerlessSpec(cloud="aws", region="us-east-1")
   )

index=pc.Index(index_name) 

from langchain_pinecone import PineconeVectorStore
docsearch=PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)