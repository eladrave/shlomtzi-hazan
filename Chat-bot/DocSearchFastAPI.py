import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.vectorstores import PGVector
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import shutil
import uvicorn
import logging
import chardet

# Load environment variables from .env file
dotenv_path = find_dotenv()
if not dotenv_path:
    raise FileNotFoundError("Could not find .env file")
load_dotenv(dotenv_path)

# Retrieve variables from the environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Provide a default model if not set

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the uploaded_files directory exists
os.makedirs("uploaded_files", exist_ok=True)

# Initialize LangChain components
embeddings = OpenAIEmbeddings(api_key=API_KEY)
vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
# Alternatively, you can use a PostgreSQL-based vector store
pg_host = os.getenv("PG_HOST", "localhost")
pg_port = os.getenv("PG_PORT", "5432")
pg_db = os.getenv("PG_DB", "postgres")
pg_user = os.getenv("PG_USER", "postgres")
pg_pass = os.getenv("PG_PASS", "")
connection_string = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

vectorstorepg = PGVector(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name="langchain_collection"
)


# Define a prompt template for similarity search
similarity_prompt = PromptTemplate(
    input_variables=["input", "docs"],
    template="Use the following documents to answer the question: {docs}\nQuestion: {input}\nAnswer:"
)

# Initialize the LLM
llm = ChatOpenAI(api_key=API_KEY, model=MODEL)
llm_chain = LLMChain(llm=llm, prompt=similarity_prompt)

@app.get("/")
def read_root():
    logging.info("Endpoint '/' called")
    return {"message": "Welcome to my app!"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    vectorstore: str = Form(...),
    groups: list[int] = Form(...)
):
    logging.info(f"Uploading file: {file.filename} to vectorstore: {vectorstore} with groups: {groups}")
    try:
        file_location = f"uploaded_files/{file.filename}"
        logging.info(f"Saving file to {file_location}")

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(file_location, "rb") as f:
            raw_data = f.read()
            detected_encoding = chardet.detect(raw_data)["encoding"]

        encoding_to_use = detected_encoding if detected_encoding else "utf-8"
        with open(file_location, "r", encoding=encoding_to_use) as f:
            content = f.read()
            try:
                metadata = {
                    "access_control_groups": groups,
                    "author": "John Doe",
                    "timestamp": "2023-10-01T12:00:00Z",
                    "filename": file.filename
                }
                if vectorstore == "chromadb":
                    logging.info("Using Chroma vector store")
                    vectorstore.add_texts([content], metadata)
                elif vectorstore == "pgvector":
                    logging.info("Using PGVector vector store")
                    vectorstorepg.add_texts([content], metadata)
                else:
                    return JSONResponse(
                        content={"error": "Invalid vectorstore type. Must be either 'chromadb' or 'pgvector'."}, 
                        status_code=400
                    )
            except Exception as e:
                logging.error(f"Error adding texts to vectorstore: {e}")
                return JSONResponse(
                    content={"error": f"Error adding texts to vectorstore: {e}"}, 
                    status_code=500
                )

        return JSONResponse(content={"message": "File uploaded successfully"}, status_code=200)
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/search")
async def search_docs(
    query: str = Form(...),
    vectorstore: str = Form(...),
    groups: list[int] = Form(...)
):
    logging.info(f"Searching documents with query: '{query}' using vectorstore: {vectorstore} and groups: {groups}")
    try:
        if vectorstore == "chromadb":
            docs = vectorstore.similarity_search(query)
        elif vectorstore == "pgvector":
            docs = vectorstorepg.similarity_search(query)
        else:
            return JSONResponse(
                content={"error": "Invalid vectorstore type. Must be either 'chromadb' or 'pgvector'."},
                status_code=400
            )

        filtered_docs = [
            d for d in docs
            if "access_control_groups" in d.metadata
            and any(g in d.metadata["access_control_groups"] for g in groups)
        ]

        if not filtered_docs:
            logging.info("No relevant documents found")
            return JSONResponse(content={"message": "No relevant documents found"}, status_code=404)

        docs_json = [{"content": d.page_content, "metadata": d.metadata} for d in filtered_docs]
        response = llm_chain.invoke({"input": query, "docs": docs_json})
        logging.info("Search successful, returning response")
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.error(f"Error searching docs: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)