from dotenv import load_dotenv  
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

load_dotenv()

# Any Model using API Key I am using Ollama Local Setup
# llm = ChatOpenAI()
# llm2 = ChatAnthropic()

model = ChatOllama(
    model="llama3.2:1b",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256,
)

# Basic Chat Integration
# response = model.invoke("What is LLM")
# print(response.content)
# print(response.response_metadata)
# print(response.usage_metadata)



