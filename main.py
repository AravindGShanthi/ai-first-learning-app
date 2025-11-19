import asyncio
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types


load_dotenv()

retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)


def output_only_response(response) -> None:
    for i in range(len(response)):
        if (
            response[i].content.parts
            and response[i].content.parts[0]
            and response[i].content.parts[0].function_response
            and response[i].content.parts[0].function_response.response
        ):
            print(f">>> {response[i].content.parts[0].function_response.response}")


def get_current_time(city: str) -> dict:
    """
    Returns the current time of a specified city

    Args:
        city: str

    Returns:
        {"city": str, "current_time": "HH:MM:SS", "status": "Fail" or "Success"}
    """

    return {"city": city, "current_time": "10:30:01", "status": "Success"}


async def main():
    root_agent = Agent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="root_agent",
        description="tells the current time in a specified city",
        instruction="You are a helpful assistant in responding the current time of a specified city. Use 'get_current_time' tool for this purpose.",
        tools={get_current_time},
    )

    runner = InMemoryRunner(agent=root_agent)

    response = await runner.run_debug("What is the current in Seoul, South korea ?")

    output_only_response(response)


asyncio.run(main())
