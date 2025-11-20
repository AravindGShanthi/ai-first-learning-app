import asyncio
import json
import re
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.google_llm import Gemini
from google.adk.models import LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool
from google.genai import types
from typing import TypedDict, Optional
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)


load_dotenv()

retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)


class UserInputs(TypedDict):
    topic: str
    learner_profile: str
    duration: int


def safe_json_loads(text):
    if text is None or text.strip() == "":
        raise ValueError("Empty string received, cannot decode JSON")

    # Remove markdown fences
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()

    try:
        return json.loads(text)
    except Exception as e:
        print("----- JSON PARSE ERROR -----")
        print("RAW TEXT:", repr(text))
        raise e


def user_input() -> UserInputs:
    input_questions = [
        "Topic of interest (eg. AI developer, Cloud technologies, Python learning)",
        "Experience level (eg. 1 year, 2 years, 10 years)",
        "Duration in days (eg. 1, 7, 10, 15, 30)",
    ]

    user_inputs = []

    for i in range(len(input_questions)):
        user_response = input(f"{input_questions[i]}: ")
        user_inputs.append(user_response)

    return {
        "topic": user_inputs[0].strip(),
        "learner_profile": user_inputs[1].strip(),
        "duration": user_inputs[2].strip(),
    }


def output_only_response(response) -> None:
    for i in range(len(response)):
        if (
            response[i].content.parts
            and response[i].content.parts[0]
            and response[i].content.parts[0].function_response
            and response[i].content.parts[0].function_response.response
        ):
            print(f">>> {response[i].content.parts[0].function_response.response}")


def exit_loop():
    """Call this function ONLY when the reviewer wants to exit the loop, indicating the lesson plan is finished and no more changes are needed."""
    return {
        "status": "approved",
        "message": "Lesson plan approved. Exiting refinement loop.",
    }


def extract_lesson_plan_response(response) -> None:
    print(f"{'='*100}")
    print(response)
    print(f"{'='*100}")


def get_current_time(city: str) -> dict:
    """
    Returns the current time of a specified city

    Args:
        city: str

    Returns:
        {"city": str, "current_time": "HH:MM:SS", "status": "Fail" or "Success"}
    """

    return {"city": city, "current_time": "10:30:01", "status": "Success"}


def basic_input_sanitize(topic: str) -> str:
    t = topic.strip()
    t = re.sub(r"\s+", " ", t)
    return t


MAX_TOPIC_LENGTH = 200
MIN_TOPIC_LENGTH = 3
ALLOWED_TOPIC_RE = re.compile(r"^[\w\s\-\:\,\.&\(\)']+$", re.UNICODE)
ALLOWED_EXPERIENCE_PATTERN = r"^\d+ year(s)?$"
BLOCKED_KEYWORDS = {
    "bomb",
    "gun",
    "how to make",
    "kill",
    "suicide",
    "child abuse",
    "porn",
    "sex",
    "hate speech",
    "terrorist",
    "attack",
    "assassinate",
}


def contains_blocked_keyword(text: str) -> bool:
    lower = text.lower()

    for kw in BLOCKED_KEYWORDS:
        if kw in lower:
            return True

    return False


def lesson_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Runs just before the LLM call

    Returns:
        - None -> proceed with model call (possibly after modifying the llm_request in-place)
        - LlmResponse -> skip model response use this response instead
    """

    prompt_test = ""

    try:
        if hasattr(llm_request, "contents") and llm_request.contents:
            if (
                hasattr(llm_request.contents[0], "parts")
                and llm_request.contents[0].parts
            ):
                if (
                    hasattr(llm_request.contents[0].parts[0], "text")
                    and llm_request.contents[0].parts[0].text
                ):
                    prompt_test = llm_request.contents[0].parts[0].text
                else:
                    prompt_test = str(llm_request)
            else:
                prompt_test = str(llm_request)
        elif hasattr(llm_request, "prompt") and llm_request.prompt:
            prompt_test = llm_request.prompt
        elif hasattr(llm_request, "messages") and llm_request.messages:
            prompt_test = " ".join([m.get("content", "") for m in llm_request.messages])
        else:
            prompt_test = str(llm_request)

    except Exception as e:
        print("Exception -> ", e)
        prompt_test = str(llm_request)

    print("prompt_test >> ", prompt_test, "")
    prompt_test = json.loads(basic_input_sanitize(prompt_test))
    print("Santized input: ", prompt_test["topic"])

    topic = prompt_test["topic"]
    experience = prompt_test["learner_profile"]
    duration = prompt_test["duration"]

    if len(topic) < MIN_TOPIC_LENGTH:
        return LlmResponse(
            text="Error: Too small for the lesson plan generator. Please provide a specific educational topic"
        )

    if len(topic) > MAX_TOPIC_LENGTH:
        return LlmResponse(
            text=f"Error: Topic too long (>{MAX_TOPIC_LENGTH} reached).  Please shorten the topic"
        )

    if not ALLOWED_TOPIC_RE.match(topic):
        return LlmResponse(
            text=f"Error: topic contains disallowed characters. Please use simple text (letters, numbers, spaces and punctuation)."
        )

    if contains_blocked_keyword(topic):
        return LlmResponse(
            text="Error: the requested topic is not permitted for this educational tool."
        )

    if not re.match(ALLOWED_EXPERIENCE_PATTERN, experience):
        return LlmResponse(
            text="Error: invalid experience error format.  Please enter the experience in this format (1 year, 2 years etc)."
        )

    if not re.match(r"^\d+$", duration):
        return LlmResponse(
            text="Error invalid duration format. Please enter the duration in integer"
        )

    return None


async def main():
    # Get user input
    user_response = user_input()

    print("User response ", user_response)

    # lesson plan generator agent
    lesson_plan_generator_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="lesson_plan_generator_agent",
        description="output the lesson plan for a given topic for a given duration with the given deadline",
        instruction="""
            You are an AI agent specialized in generating structured lesson plans. Your task is to create a daily lesson plan based on a user-provided topic, the user's experience in that topic, and a specified duration.

            Instructions for generating the lesson plan:

            1. Inputs (from user):
                - Topic: A string representing the subject or area to create a lesson plan for.
                - Experience: Very specific to the given topic (e.g., 1 year, 2 years, 20 years). Use this to adjust the complexity and depth of the content.
                - Duration: Number of days to generate the lesson plan for. Each day should cover only one main topic/concept.

            2. Outputs:
                - An array of strings, where each string represents the lesson topic or concept for that day.
                - The lesson plan should not exceed the given duration.
                - Structure the topics so that they are progressive, considering the user’s experience level.

            3. Formatting Rules:
                - Output must be a valid array of strings: ["Day 1 Topic", "Day 2 Topic", ...]
                - Each entry should be concise but descriptive of the concept to be taught on that day.

            4. Sample input/output examples
                example 1: 
                    Input: {
                        "Topic": "Python Programming",
                        "Experience": "1 year",
                        "Duration": 3
                    },
                    Output: [
                        "Day 1: Advanced Data Structures in Python (Lists, Dictionaries, Sets)",
                        "Day 2: Object-Oriented Programming Concepts and Classes",
                        "Day 3: File Handling and Exception Management"
                    ]

                example 2:
                    Input: {
                        "Topic": "Digital Marketing",
                        "Experience": "2 years",
                        "Duration": 4
                    },
                    Output: [
                        "Day 1: Advanced SEO Techniques and Keyword Research",
                        "Day 2: Paid Advertising Strategies (Google Ads, Social Media Ads)",
                        "Day 3: Content Marketing Optimization",
                        "Day 4: Analytics and Performance Measurement"
                    ]

                example 3:
                    Input:
                    {
                        "Topic": "Data Science",
                        "Experience": "5 years",
                        "Duration": 2
                    }
                    Output:
                    [
                        "Day 1: Advanced Machine Learning Algorithms (XGBoost, Random Forest)",
                        "Day 2: Model Deployment and Monitoring in Production"
                    ]

        """,
        output_key="generated_plan",
        before_model_callback=lesson_before_model_callback,
    )

    # lesson plan reviewer agent
    lesson_plan_reviewer_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="lesson_plan_reviewer_agent",
        description="output the feedback after reviewing the lesson plan generated by lesson plan generatory agent",
        instruction="""
            You are an expert Lesson Plan Reviewer with 10+ years of experience in instructional design, curriculum sequencing, and educational pedagogy. Your task is to review lesson plans generated by another agent and provide actionable feedback.

            Inputs (from the lesson plan generator agent):
            - Topic: The subject or area the lesson plan is about.
            - Experience: The user’s experience in the topic (e.g., 1 year, 5 years, 20 years). Use this to evaluate if the complexity and depth of the lessons are appropriate.
            - Duration: Number of days for which the lesson plan is intended. Ensure the plan covers only this duration and one topic/concept per day.
            - Generated Plan: A list of strings representing the daily topics/concepts, in order.

            Your Tasks:
            1. Review the Generated Plan carefully. Consider:
            - Relevance of each day’s topic to the main Topic.
            - Complexity relative to the user’s Experience.
            - Logical progression of topics over the given Duration.
            - Adherence to the rule of one topic/concept per day.

            2. Provide feedback in a general, structured format, including:
            - Whether the plan is well-structured, progressive, and suitable for the user’s experience.
            - If any topics are out of order, redundant, or missing.
            - Suggestions to refine or replace topics to improve the learning experience.

            3. Output Rules:
                If the plan is excellent and needs no changes:
                    - return "APPROVED";

                Otherwise, output a structured feedback object in JSON or text, e.g.:

                {
                    "feedback": "The lesson plan is mostly well-structured, but Day 2 topic seems too basic for the user's 5 years experience. Consider replacing it with a more advanced topic on X.",
                    "issues": [
                    {"day": 2, "issue": "Too basic for experience level"},
                    {"day": 4, "issue": "Missing advanced topic on Y"}
                    ]
                }

                - Feedback must be clear, actionable, and concise so the refiner agent can directly use it to improve the plan.

            Important Notes:
            - Always consider the user’s experience, topic, and duration in your evaluation.
            - Never suggest topics beyond the specified Duration.
            - Maintain professional, educational tone in your feedback.

            """,
        output_key="reviewer_output",
    )

    lesson_plan_refiner_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="lesson_plan_refiner_agent",
        description="",
        instruction="""
            You are an AI Lesson Plan Refiner. Your task is to take the lesson plan generated by the Lesson Plan Generator and refine it based on feedback from the Lesson Plan Reviewer Agent.

            1. Inputs:

            1.1. Generated Plan: The original lesson plan array from the generator agent (["Day 1 Topic", "Day 2 Topic", ...]).
            1.2. Reviewer feedback: Feedback provided by the Lesson Plan Reviewer Agent. It may include:
                - Specific issues with certain days/topics.
                - Suggestions for replacing, reordering, or refining topics.
                - General recommendations for improving progression or suitability for the user’s experience.
                - User Inputs:
                    - Topic: The subject area.
                    - Experience: User’s experience level (e.g., 1 year, 5 years, 20 years).
                    - Duration: Number of days for the lesson plan.

            Your Task:

            1. Carefully analyze the Reviewer Feedback.
            - If the Reviewer feedback output is "APPROVED" exit the loop by calling exit_loop() function tool immediately

            2. Refine the Generated Plan so that:
            - All issues mentioned in the feedback are addressed.
            - Topics are appropriate to the user’s experience.
            - Progression of topics over the duration is logical.
            - Only one topic/concept per day is included.

            3. If the feedback indicates no issues, do not make changes; instead, return exit_loop().

            4. Otherwise, produce a new refined plan as an array of strings in the same format as the generator:

            ["Day 1: Refined Topic", "Day 2: Refined Topic", ...]


            Output Rules:

            - Only output the refined lesson plan array or exit_loop().
            - Ensure the plan strictly adheres to the user’s topic, experience, and duration.
            - Do not add additional days or topics beyond the given duration.
            - Maintain a clear and progressive flow of concepts suitable for the user’s experience level.

            Available Tools
            - Exit_loop() - To exit the feedback loop

            Example Workflow:

            Input:

            {
            "Generated Plan": ["Day 1: Basic Python Lists", "Day 2: OOP Basics", "Day 3: File Handling"],
            "Reviewer Feedback": {
                "feedback": "Day 1 is too basic for the user's 5 years experience. Consider advanced data structures.",
                "issues": [{"day": 1, "issue": "Too basic for experience"}]
            },
            "User Inputs": {
                "Topic": "Python Programming",
                "Experience": "5 years",
                "Duration": 3
            }
            }


            Output:

            ["Day 1: Advanced Data Structures in Python (Lists, Dictionaries, Sets)", 
            "Day 2: Object-Oriented Programming Concepts and Classes", 
            "Day 3: File Handling and Exception Management"]
        """,
        output_key="refiner_output",
        tools=[FunctionTool(exit_loop)],
    )

    lesson_plan_refinement_loop = LoopAgent(
        name="LessonRefinementLoop",
        sub_agents=[lesson_plan_reviewer_agent, lesson_plan_refiner_agent],
        max_iterations=3,  # Prevents infinite loops
    )

    root_agent = SequentialAgent(
        name="LessonPlanPipeline",
        sub_agents=[lesson_plan_generator_agent, lesson_plan_refinement_loop],
    )

    runner = InMemoryRunner(agent=root_agent, plugins=[LoggingPlugin()])

    response = await runner.run_debug(json.dumps(user_response))

    print("\n\n\n Output:\n")

    print("\n\n\n Response Start")
    print(response)
    print("\n\n\n Response End")

    final_plan = None
    for step in response:
        if hasattr(step, "content") and step.content.parts:
            for part in step.content.parts:
                if hasattr(part, "text"):
                    try:
                        parsed = safe_json_loads(part.text)
                        if (
                            isinstance(parsed, list)
                            and parsed
                            and parsed[0].startswith("Day ")
                        ):
                            final_plan = parsed
                    except:
                        pass

    if final_plan:
        print("\nFINAL LESSON PLAN:")
        for day in final_plan:
            print(day)

    else:
        print("No plan generated.")


if __name__ == "__main__":
    asyncio.run(main())
    print("Program done")
