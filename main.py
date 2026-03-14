from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama

load_dotenv()

model = ChatOllama(
    model="llama3.2:1b",
    temperature=0.3,
    num_predict=80,
)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]


structured_llm = model.with_structured_output(ResearchResponse)

try:
    result = structured_llm.invoke(
        "What is the capital of France?"
    )

    print("Topic:", result.topic)
    print("Summary:", result.summary)
    print("Sources:", result.sources)
    print("Tools:", result.tools)

except Exception as e:
    print("Error:", e)