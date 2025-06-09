from agno.agent import Agent
from agno.tools.tavily import TavilyTools
from agno.tools.wikipedia import WikipediaTools
from agno.models.groq import Groq
from pydantic import BaseModel, Field
from typing import List
from rich.pretty import pprint

class Template(BaseModel):
    ans: str = Field(..., description="Provide answer for the question asked. If you don't know the answer, say 'I don't know'.")

agent = Agent(
    model = Groq(id="llama-3.1-8b-instant",api_key="******"),
    description="An agent that can answer questions.",
    markdown=False,
    response_model=Template,
    #use_json_mode=True,
    )

agent.print_response("What is the capital of France?")