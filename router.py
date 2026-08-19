import re
from datetime import datetime

from llm import classify_route

GREETINGS = {
    "hi",
    "hie",
    "hello",
    "hey",
    "hey there",
    "how are you",
    "how r you",
    "whats up",
    "what's up",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "thanks",
    "thank you",
    "thank you so much",
    "bye",
    "goodbye",
    "see you",
    "ok",
    "okay",
    "namaste",
    "yo",
}


def is_greeting(question):
    text = re.sub(r"[!?.]+$", "", question.strip().lower()).strip()
    return text in GREETINGS


def get_current_time_context():
    now = datetime.now()
    return (
        f"Current local date and time on this computer: "
        f"{now.strftime('%A, %d %B %Y, %I:%M:%S %p')}"
    )


def choose_mode(question):
    """
    Fast-path greetings, then LLM classification.
    Returns: CHAT | TIME | WEB | DOC
    """
    if is_greeting(question):
        return "CHAT"
    return classify_route(question)
