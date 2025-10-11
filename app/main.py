import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from pinecone import Pinecone, ServerlessSpec

import json

from fastapi import FastAPI

load_dotenv()

API_KEY = os.getenv('API_KEY')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pinecone Database
#PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
#pc = Pinecone(api_key= PINECONE_API_KEY )

# LangSmith tracking
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"

## Pompt Template


prompt=ChatPromptTemplate.from_messages([
    ("system","Your are a helpful assistant. Please response to the user queries"),
    ("user","Question: {question}"),
])





llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)
output_parser=StrOutputParser()

chain=prompt| llm | output_parser




## Testing Pinecone



#client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)


#print(API_KEY)
#app = FastAPI()







