from dotenv import load_dotenv  
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent

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

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources: list[str]
    tools: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text.

            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a research assistant.",
    response_format=ResearchResponse
)


raw_response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ]
    }
)

print(raw_response)