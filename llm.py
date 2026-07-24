import os

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)


def _chat(prompt):
    response = client.chat.completions(
        model="sarvam-30b",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        return ""
    return str(content).strip()


def classify_route(question):
    """
    LLM router — no keyword lists.
    Returns: DOC | WEB | TIME
    """
    prompt = f"""
You are a routing classifier for a Q&A app.

Choose exactly ONE label for the user question:

DOC  = question is about the user's uploaded project document
       (PMS, Nexus, Apparatus, project modules, requirements,
        roles, APIs, features of that project PDF)
WEB  = question needs live public/internet knowledge
       (news, politics, sports, world facts, people, current events)
TIME = question asks for the current time or today's date on this computer

User question:
{question}

Reply with ONLY one word: DOC or WEB or TIME
"""
    result = _chat(prompt).upper()

    if "TIME" in result:
        return "TIME"
    if "WEB" in result:
        return "WEB"
    if "DOC" in result:
        return "DOC"

    # Safe default for this project app
    return "DOC"


def generate_answer(question, context):
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the provided context.
Be direct and clear.
If the context contains the answer, give it in 1-3 short sentences.
Do not say "None".
If the answer is truly not in the context, reply exactly:
I don't know based on the provided context.

Context:
{context}

Question:
{question}
"""

    answer = _chat(prompt)
    if not answer or answer.lower() == "none":
        return "I don't know based on the provided context."
    return answer
