from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
project_name = os.getenv("LANGCHAIN_PROJECT")

# Environment setup
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = project_name
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer clearly and concisely."),
    ("user", "{question}")
])

# Streamlit UI
st.title("Chatbot using Groq AI + LangChain")

input_text = st.text_input("Enter your question:")

# Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Run chatbot
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)