from flask import Flask,render_template,request
from dotenv import load_dotenv
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.prompt import system_prompt
import os

app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

embeddings=download_embeddings()
index_name="medical-chatbot"

doc_search=PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever=doc_search.as_retriever(search_type="similarity", search_kwargs={"k":3})
llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9) 

prompt=ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)





