import asyncio
import json
import re
import uuid
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.google_llm import Gemini
from google.adk.models import LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools.tool_context import ToolContext
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

    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1]).strip()

    try:
        return json.loads(text)
    except Exception as e:
        print("----- JSON PARSE ERROR -----")
        print("RAW TEXT:", repr(text))
        # raise e


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


def user_concept_selection(available_topics: list[str]) -> str:
    while True:
        try:
            user_input = int(
                input(
                    "Please select the concept to learn. Eg. If you want to select day 5 please enter 5: \t"
                )
            )

            if user_input < 0:
                return "EXIT"

            print(
                "User input => ",
                user_input,
                user_input > 0,
                user_input <= len(available_topics),
                type(user_input),
            )
            if user_input > 0 and user_input <= len(available_topics):
                return available_topics[user_input - 1]
            else:
                print("Invalid user input.  Please select again")
                continue
        except Exception as e:
            print("Invalid user input.  Please select again", e)
            continue


def exit_inner_loop(tool_context: ToolContext):
    """Call this function ONLY when the reviewer wants to exit the loop, indicating the lesson/content plan is finished and no more changes are needed."""

    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")

    return {
        "status": "approved",
        "message": "Lesson/content plan approved. Exiting refinement loop.",
    }


def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the reviewer wants to exit the loop, indicating the lesson/content plan is finished and no more changes are needed."""

    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True

    return {
        "status": "approved",
        "message": "Lesson/content plan approved. Exiting refinement loop.",
    }


def human_feedback_input(
    lesson_plans: list[str],
    tool_context: ToolContext,
    human_input: Optional[str] = "",
) -> dict:
    """
    Ask for human feedback.  This function presents the lesson plans to the end user to get the feedback from the input.

    Args:
        lesson_plans: this is the final generated plans from the refinement loop
        human_input: this is the human provided feedback input

    Returns:
        Dictionary with user approval
    """

    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Please review and approve the lesson plans:\n {",\n".join(lesson_plans)}",
            payload={"lesson_plans": lesson_plans},
        )

        return {
            "status": "pending",
            "message": f"Request for lesson plan approval is pending, the lesson plans:\n {",\n".join(lesson_plans)}",
        }

    tc = tool_context.tool_confirmation

    confirmed = getattr(tc, "confirmed", None)

    feedback_text = human_input or ""

    if not feedback_text and hasattr(tool_context, "user_content"):
        for part in getattr(tool_context.user_content, "parts", []):
            text = getattr(part, "text", None)
            if text:
                print("TEXT ===> ", text)
                try:
                    feedback_json = json.loads(text)
                    if "human_input" in feedback_json:
                        feedback_text = feedback_json["human_input"]
                        break
                except Exception:
                    continue

    if confirmed:
        return {"status": "approved", "feedback": feedback_text}
    else:
        return {"status": "rejected", "feedback": feedback_text or "rejected"}


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

content_map = {}


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

    prompt_test = json.loads(basic_input_sanitize(prompt_test))

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


def extract_latest_output(events):
    """
    Universal extractor:
    - Extracts the last valid JSON object found anywhere in the event logs.
    - If no JSON exists, returns the last "APPROVED" (if present).
    - Handles reviewer/refiner/generator outputs equally.
    """

    last_json = None
    last_approved = None

    for event in events:
        if not hasattr(event, "content") or not event.content:
            continue

        parts = getattr(event.content, "parts", [])
        if not parts:
            continue

        for part in parts:
            if part is None:
                continue

            text = getattr(part, "text", None)
            if isinstance(text, str):
                stripped = text.strip()

                if stripped == "APPROVED":
                    last_approved = "APPROVED"

                json_block = re.search(r"```json\s*(.*?)```", stripped, re.DOTALL)
                if json_block:
                    raw_json = json_block.group(1).strip()
                    try:
                        parsed = json.loads(raw_json)
                        last_json = parsed
                    except Exception:
                        pass

            func_resp = getattr(part, "function_response", None)
            if func_resp:
                response_data = func_resp.response
                if (
                    isinstance(response_data, dict)
                    and response_data.get("status") == "approved"
                ):
                    last_approved = "APPROVED"

    if last_json is not None:
        return last_json

    if last_approved is not None:
        return "APPROVED"

    return None


def check_for_approval(events):
    """
    check if events contain an approval request

    Return:
        dict with approval details or None
    """

    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")


def create_approval_response(approval_info: dict, user_feedback: dict):
    """Create approval response message"""
    confirmed_bool = bool(user_feedback.get("status", False))
    feedback_text = str(user_feedback.get("feedback", ""))

    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": confirmed_bool},
    )

    parts = [types.Part(function_response=confirmation_response)]
    if feedback_text:
        parts.append(types.Part(text=json.dumps({"human_input": feedback_text})))

    return types.Content(role="user", parts=parts)


CONTENT_GENERATOR_OUTPUT = "generated_content"
CONTENT_REVIEWER_OUTPUT = "reviewer_content"


async def content_generator(concept: str) -> str:
    content_generator_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="content_generator_agent",
        description="output the generated content for a given topic/concept.",
        instruction="""
            You are an expert AI Content Generator for educational purposes. Your task is to generate **high-quality, concise, and factually accurate content** for a single lesson concept or topic provided by the user. You will receive:

            - topic/concept: The lesson topic to generate content for.

            Your responsibilities:

            1. Generate content strictly in JSON format with the following fields:

            {{
            "title": "<Title of the lesson, concise and informative>",
            "description": "<A short summary of the lesson, 2-3 sentences>",
            "detailed_explanation": "<Detailed explanation of the concept/topic. Include examples, code samples, diagrams (if applicable), or step-by-step explanation. Keep it concise, easy to understand, and factually accurate.>",
            "external_resources": {{
                "free": [
                {{
                    "title": "<Course/Resource title>",
                    "url": "<URL of the resource>",
                    "rating": "<Rating or reviews>",
                    "price": "<Price if any, else 0>",
                    "value_for_money": "<High/Medium/Low>"
                }}
                ... (maximum 5 items)
                ],
                "paid": [
                {{
                    "title": "<Course/Resource title>",
                    "url": "<URL of the resource>",
                    "rating": "<Rating or reviews>",
                    "price": "<Price in USD or local currency>",
                    "value_for_money": "<High/Medium/Low>"
                }}
                ... (maximum 5 items)
                ]
            }}
            }}

            2. Requirements and Rules:

            - **Strict JSON only:** Do not include any text outside JSON. Invalid JSON is unacceptable.
            - **Fact-checking:** Ensure all explanations are accurate; do not hallucinate. Use Google search tools if required.
            - **Content style:** Short, clear, and informative. Suitable for learners with basic to intermediate knowledge.
            - **Code/Example inclusion:** Include code snippets or examples where relevant. Escape all curly braces in code snippets with double braces (e.g., `{{` and `}}`).
            - **External resources:** List only the top 5 free and top 5 paid resources. Sort by reviews, price, and value for money. Use reputable sources only.
            - **No filler content:** Avoid verbose introductions or unnecessary commentary.
            - **JSON validation:** Ensure all strings are properly quoted, arrays and objects are well-formed, and there are no trailing commas.
            - **Avoid subjective statements** unless sourced from external reviews.

            3. Example structure (do NOT copy content):

            {{
            "title": "Introduction to Python Functions",
            "description": "Learn the basics of Python functions and how to use them effectively.",
            "detailed_explanation": "A Python function is a reusable block of code that performs a specific task. Example:\\n```python\\ndef greet(name):\\n    return f\"Hello, name!\"\\n```\\nFunctions allow modularity and easier maintenance of code.",
            "external_resources": {{
                "free": [
                {{ "title": "Python Official Tutorial", "url": "https://docs.python.org/3/tutorial/", "rating": "4.8/5", "price": "0", "value_for_money": "High" }},
                ...
                ],
                "paid": [
                {{ "title": "Complete Python Bootcamp", "url": "https://www.udemy.com/course/complete-python-bootcamp/", "rating": "4.7/5", "price": "$12.99", "value_for_money": "High" }},
                ...
                ]
            }}
            }}

            Your output must **strictly follow the JSON structure above** and include both free and paid resources. Focus on concise, accurate, and high-quality educational content.
            """,
        output_key=CONTENT_GENERATOR_OUTPUT,
    )

    content_reviewer_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="content_reviewer_agent",
        description="output the feedback if any in a predefined JSON output format otherwise 'APPROVED' as output",
        instruction="""
            You are an expert Lesson Content Reviewer with 10+ years of experience in educational content quality and instructional design.

            Your task is to review the JSON output generated by the Content Generator Agent.

            You will receive:
            - topic/concept: The lesson topic
            - generated_content: JSON content generated for the lesson

            Responsibilities:

            1. Review the content for:
            - **Accuracy**: Verify all facts, examples, and code.
            - **Clarity & Conciseness**: Content must be easy to understand and not verbose.
            - **Completeness**: JSON must have all required fields (title, description, detailed_explanation, external_resources with top 5 free/paid resources).
            - **External Resources**: Must be reputable, sorted by reviews, price, and value for money.
            - **JSON Validity**: Ensure well-formed JSON; no trailing commas.
            - **Code Examples**: Check if examples or code snippets are correct.

            2. Output Rules:
            - If the content is perfect and requires **no changes**, your output must be the single string:
                "APPROVED" and do not return anything.
            - If there are issues, output **strict JSON only** in the following format:

            {
            "review_status": "REJECTED",
            "comments": [
                "<Actionable feedback comment 1>",
                "<Actionable feedback comment 2>",
                ...
            ]
            }

            3. Important:
            - Do not output any text outside the JSON or the single string "APPROVED".
            - Feedback comments should be **specific and actionable**, e.g., "Fix the code example for integer division", "Add one more reputable free resource", "Clarify the explanation of NoneType".

            Remember: Your output will control whether the content goes back to the Refiner Agent or exits the loop.
            """,
        output_key=CONTENT_REVIEWER_OUTPUT,
    )

    content_refiner_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="content_refiner_agent",
        description="output the refined output for the content reviewer to review otherwise exit the loop",
        instruction="""
            You are an expert AI Content Refiner. Your job is to improve lesson content generated by the Content Generator Agent based on feedback from the Reviewer Agent.

            You will receive:
            - topic/concept: The lesson topic
            - generated_content: The JSON content generated by the Content Generator
            - reviewer_content: Either the string "APPROVED" or a JSON object with actionable feedback

            Your responsibilities:

            1. If `reviewer_content` is the string "APPROVED":
                - Call the `exit_loop` tool to finalize the content.
                - Do not return any other output.

            2. If `reviewer_content` contains actionable feedback (JSON with comments):
                - Refine the `generated_content` according to the reviewer comments.
                - Ensure improvements are specific, concise, and factually accurate.
                - Maintain **strict JSON structure**:
            {
            "title": "<Refined lesson title>",
            "description": "<Refined 2-3 sentence summary>",
            "detailed_explanation": "<Refined detailed explanation, including examples or code snippets>",
            "external_resources": {
                "free": [ ... up to 5 items ... ],
                "paid": [ ... up to 5 items ... ]
            }
            }

            3. Rules:
                - Fact-check all content; avoid hallucinations.
                - Correct any errors or inconsistencies noted in the reviewer feedback.
                - Include or improve code snippets or examples as needed.
                - Ensure external resources are relevant, reputable, and within top 5 free/paid.
                - Output only the refined JSON; do not include any extra commentary or text.

            Remember: Your output will be sent back to the Content Reviewer Agent for another review cycle if changes were made, or will exit the loop if approved.
            """,
        output_key=CONTENT_GENERATOR_OUTPUT,
        tools=[FunctionTool(exit_loop)],
    )

    content_refinement_loop = LoopAgent(
        name="ContentRefinementLoop",
        sub_agents=[content_reviewer_agent, content_refiner_agent],
        max_iterations=5,  # Prevents infinite loops
    )

    root_agent = SequentialAgent(
        name="ContentPlanPipeline",
        sub_agents=[content_generator_agent, content_refinement_loop],
    )

    runner = InMemoryRunner(agent=root_agent, plugins=[LoggingPlugin()])

    events = await runner.run_debug(concept)

    final_output = []

    for event in events:
        if event.is_final_response() and event.content:
            print(f"{'=' * 80}")
            if hasattr(event.content, "parts") and event.content.parts:
                parts = []
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text.strip())
                        print(" > ", part.text.strip())
                joined_parts = "".join(parts)
                parsed_output = safe_json_loads(joined_parts)
                print("Parsed_output", parsed_output)
                if parsed_output is not None and "title" in parsed_output:
                    final_output = parts
            print(f"{'=' * 80}")

    joined_values = "".join(final_output)

    print("XXXX" * 10)
    print(joined_values)
    print("XXXX" * 10)

    return final_output


events = []


async def ai_executor(
    runner: Runner,
    user_id: str,
    session_id: str,
    query_content: str,
    invocation_id: Optional[str] = None,
):
    dummy_events = []
    if invocation_id:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=query_content,
            invocation_id=invocation_id,
        ):
            dummy_events.append(event)
            events.append(event)
    else:
        return

    approval_info = check_for_approval(dummy_events)

    if approval_info:
        approved_input = (
            input("Approve the lesson plan? (y/n) [y = approve]: ").strip().lower()
        )

        approved_bool = approved_input in ("y", "yes", "approve", "approved")

        feedback_text = input(
            "Optional feedback (enter any comments to pass to the refiner agent): "
        ).strip()

        query_content = create_approval_response(
            approval_info=approval_info,
            user_feedback={
                "status": approved_bool,
                "feedback": feedback_text,
            },
        )

        await ai_executor(runner, user_id, session_id, query_content, invocation_id)
    else:
        return


async def main():
    user_response = user_input()

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

    lesson_plan_reviewer_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="lesson_plan_reviewer_agent",
        description="output the feedback after reviewing the lesson plan generated by lesson plan generatory agent",
        instruction="""
            You are a Senior Lesson Plan Reviewer with expertise in instructional design and curriculum progression.
            Your job is to evaluate and provide mandatory revisions for lesson plans generated by another AI.

            You will be given the following inputs:
            - TOPIC: What the lesson plan is about.
            - EXPERIENCE: The learner’s experience level. Use this to judge complexity and pacing.
            - DURATION: Number of days the plan must span. Ensure EXACT match.
            - GENERATED_PLAN: A list of daily topics/concepts.
            - HUMAN_INPUT: Direct feedback from the user. This input is ALWAYS prioritized above all other considerations. If HUMAN_INPUT contradicts the GENERATED_PLAN, HUMAN_INPUT MUST override.

            Your Evaluation Criteria:
            1. TOPIC RELEVANCE:
            - Every day must stay aligned with the main topic.

            2. HITL FEEDBACK MUST BE APPLIED:
            - HUMAN_INPUT suggestions are NON-NEGOTIABLE.
            - If HUMAN_INPUT affects flow or structure, recommend restructuring accordingly.

            3. EXPERIENCE ALIGNMENT:
            - Ensure topic difficulty progression matches user experience level.

            4. LOGICAL PROGRESSION:
            - Each day should build on the previous.
            - One concept/topic per day only.

            Your Output Rules:
            - FIRST: Determine whether the plan fully satisfies:
            - Relevance
            - Logical progression
            - Level-appropriateness
            - DURATION match
            - FULL incorporation of HUMAN_INPUT

            - If ALL criteria are satisfied:
            Return EXACTLY:
            APPROVED

            - Otherwise:
            Return a JSON object with this STRICT structure:

            {
                "review_status": "NEEDS_IMPROVEMENT",
                "feedback": "Short general summary of required improvements.",
                "required_changes": [
                {
                    "day": <day_number or null if applies to entire plan>,
                    "issue": "Short description of the problem",
                    "required_action": "Specific instruction on what must change"
                }
                ]
            }

            Guidelines:
            - Be specific: identify exact days and topics needing change.
            - Use clear instructions that the Refiner can directly implement.
            - Never suggest adding more days than specified in DURATION.
            - Never ignore HUMAN_INPUT.
            - Maintain a professional and constructive educational tone.

            Compliance:
            Failure to incorporate HUMAN_INPUT or incorrectly marking a flawed plan as APPROVED
            is considered a critical error.
        """,
        output_key="reviewer_output",
    )

    lesson_plan_refiner_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="lesson_plan_refiner_agent",
        description="",
        instruction="""
            You are an AI Lesson Plan Refiner. Your role is to revise lesson plans using the Reviewer's feedback and Human Input (HITL). Your output is directly consumed by an automated loop — therefore precision and rule compliance are mandatory.

            You will receive:
            - GENERATED_PLAN: Array of lesson plan topics, one per day.
            - REVIEWER_OUTPUT: Structured output from Reviewer containing:
                - review_status: "APPROVED" or "NEEDS_IMPROVEMENT"
                - required_changes: List of specific modifications to apply
            - HUMAN_INPUT: Direct user feedback. This is the HIGHEST PRIORITY and MUST be included exactly, even if:
                - It alters the structure
                - It disrupts logical flow
                - It contradicts the Reviewer's plan

            Your Refinement Rules:

            1. Loop Exit Condition
            If REVIEWER_OUTPUT.review_status == "APPROVED":
            - You MUST call the `exit_loop()` tool.
            - Output NOTHING else. No quotes. No explanations.

            2. Refinement Condition
            If REVIEWER_OUTPUT.review_status == "NEEDS_IMPROVEMENT":
            - You MUST revise the plan so that:
                ✔ All required_changes are fully applied
                ✔ All HUMAN_INPUT feedback is explicitly incorporated
                ✔ Topics remain aligned with USER Topic and Duration
                ✔ Each day has ONE topic only
                ✔ Complexity matches Experience level
                ✔ No extra days are added

            3. Output Format
            - Output ONLY the final refined lesson plan as:
            ["Day 1: ...", "Day 2: ...", ...]
            - No JSON wrapper
            - No narrative text
            - No explanations before or after

            4.  Behavioral Restrictions
            - Never ignore HUMAN_INPUT — it overrides all other logic
            - Never mention review status or the Reviewer in the output
            - Never instruct, only rewrite
            - Never propose more days than Duration
            - Never produce empty fields or placeholders

            5.  Compliance Rules
            Failure to:
            - apply required changes,
            - include HUMAN_INPUT fully,
            - or exit loop properly when approved,
            is considered a critical task failure.

            Final Summary:
            If APPROVED → call exit_loop()
            If NEEDS_IMPROVEMENT → return revised plan array only
        """,
        output_key="generated_plan",
        tools=[FunctionTool(exit_inner_loop)],
    )

    human_feedback_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="human_feedback_agent",
        description="human review and approve the final refined lesson plan",
        instruction="""
            You are an human feedback request agent

            1. Use the human_feedback_input tool with the latest final generated lesson plan and human provided feedback if available or else use None as inputs, if human status is "approved" call the exit_loop tool to exit the loop.
            2. If the human feedback status is "pending", inform the user that the approval is pending.
            3. If the human feedback status is "rejected" then continue with the loop.
        """,
        tools=[FunctionTool(exit_loop), FunctionTool(human_feedback_input)],
        output_key="human_input",
    )

    lesson_plan_refinement_loop = LoopAgent(
        name="LessonRefinementLoop",
        sub_agents=[lesson_plan_reviewer_agent, lesson_plan_refiner_agent],
        max_iterations=1,  # Prevents infinite loops
    )

    human_feedback_loop = LoopAgent(
        name="HumanFeedbackLoop",
        sub_agents=[lesson_plan_refinement_loop, human_feedback_agent],
        max_iterations=5,
    )

    root_agent = SequentialAgent(
        name="LessonPlanPipeline",
        sub_agents=[lesson_plan_generator_agent, human_feedback_loop],
    )

    lesson_plan_app = App(
        name="lesson_plan_coordinator",
        root_agent=root_agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
        plugins=[LoggingPlugin()],
    )

    session_service = InMemorySessionService()

    runner = Runner(app=lesson_plan_app, session_service=session_service)

    session_id = f"lesson_{uuid.uuid4().hex[:8]}"
    user_id = "test_user"

    await session_service.create_session(
        app_name="lesson_plan_coordinator", user_id=user_id, session_id=session_id
    )

    query_content = types.Content(
        role="user", parts=[types.Part(text=json.dumps(user_response))]
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=query_content,
    ):
        events.append(event)

    approval_info = check_for_approval(events)

    if approval_info:
        approved_input = (
            input("Approve the lesson plan? (y/n) [y = approve]: ").strip().lower()
        )

        approved_bool = approved_input in ("y", "yes", "approve", "approved")

        feedback_text = input(
            "Optional feedback (enter any comments to pass to the refiner agent): "
        ).strip()

        query_content = create_approval_response(
            approval_info=approval_info,
            user_feedback={
                "status": approved_bool,
                "feedback": feedback_text,
            },
        )

        await ai_executor(
            runner, user_id, session_id, query_content, approval_info["invocation_id"]
        )

    final_output = []

    for event in events:
        if event.is_final_response() and event.content:
            if hasattr(event.content, "parts") and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        try:
                            parts = safe_json_loads(part.text)
                            if isinstance(parts, list) and all(
                                isinstance(x, str) and x.startswith("Day ")
                                for x in parts
                            ):
                                final_output = parts
                        except:
                            pass

    if final_output:
        for day in final_output:
            print(day)

        while True:
            concept_tobe_used = user_concept_selection(final_output)
            if concept_tobe_used == "EXIT":
                break

            if content_map.get(concept_tobe_used, ""):
                print(
                    "Content Already exists Found => ",
                    content_map.get(concept_tobe_used, ""),
                )
            else:
                print("Storing content")
                content_output = await content_generator(concept_tobe_used)
                content_map[concept_tobe_used] = content_output

    else:
        print("No plan generated.")


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(
    #     content_generator(
    #         "Advanced Data Structures and the Collections Module in python"
    #     )
    # )
    print("Program done")
