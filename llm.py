import os

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)


def _chat(prompt):
    response = client.chat.completions(
        model="sarvam-105b",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        return ""
    return str(content).strip()


def classify_route(question):
    """
    LLM router — no keyword lists.
    Returns: CHAT | DOC | WEB | TIME
    """
    prompt = f"""
You are a routing classifier for a Q&A app.

Choose exactly ONE label for the user question:

CHAT = greeting or small talk
       (hi, hello, hey, how are you, thanks, bye, good morning)
DOC  = question is about the user's uploaded project document
       (PMS, Nexus, Apparatus, project modules, requirements,
        roles, APIs, features of that project PDF)
WEB  = question needs live public/internet knowledge
       (news, politics, sports, world facts, people, current events)
TIME = question asks for the current time or today's date on this computer

User question:
{question}

Reply with ONLY one word: CHAT or DOC or WEB or TIME
"""
    result = _chat(prompt).upper()

    if "TIME" in result:
        return "TIME"
    if "CHAT" in result:
        return "CHAT"
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
and give the answer in the same language as the question.
If the question is in English, give the answer in English.
If the question is in Hindi, give the answer in Hindi.
If the question is in Marathi, give the answer in Marathi.
If the question is in Gujarati, give the answer in Gujarati.
If the question is in Tamil, give the answer in Tamil.
If the question is in Telugu, give the answer in Telugu.
If the question is in Kannada, give the answer in Kannada.
If the question is in Malayalam, give the answer in Malayalam.

Context:
{context}

Question:
{question}
"""

    answer = _chat(prompt)
    if not answer or answer.lower() == "none":
        return "I don't know based on the provided context."
    return answer


def generate_chat_reply(question):
    prompt = f"""
You are a friendly assistant talking to Siva.
The user's name is Siva. Always use that name.

If they say hi/hello/hey: greet them by name, short and warm.
If they ask how are you: say you are fine, then ask how Siva is.
If they say thanks/bye: reply politely using the name Siva.
Keep it to 1-2 sentences.
Match the user's language (English, Hindi, Telugu, Tamil, etc.).

User:
{question}
"""
    answer = _chat(prompt)
    if not answer:
        return "Hi Siva, how can I help you?"
    return answer
