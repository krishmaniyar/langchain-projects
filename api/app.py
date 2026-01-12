from fastapi import FastAPI
from langchain_core.prompts  import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

load_dotenv()

# Environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="LangChain Groq Server",
    version="1.0",
    description="A simple API Server using Groq"
)

# -----------------------
# Groq LLM
# -----------------------
groq_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# -----------------------
# Ollama LLM (local)
# -----------------------
ollama_llm = Ollama(model="llama2")

# -----------------------
# Prompts
# -----------------------
prompt1 = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words"
)

prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5 years old child with 100 words"
)

# -----------------------
# Routes
# -----------------------

# Direct Groq chat endpoint
add_routes(
    app,
    groq_model,
    path="/groq"
)

# Essay using Groq
add_routes(
    app,
    prompt1 | groq_model,
    path="/essay"
)

# Poem using Ollama
add_routes(
    app,
    prompt2 | ollama_llm,
    path="/poem"
)

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
