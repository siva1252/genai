from datetime import datetime

from llm import classify_route


def get_current_time_context():
    now = datetime.now()
    return (
        f"Current local date and time on this computer: "
        f"{now.strftime('%A, %d %B %Y, %I:%M:%S %p')}"
    )


def choose_mode(question):
    """
    Smart router using LLM classification (no hardcoded keyword hints).
    Returns: TIME | WEB | DOC
    """
    return classify_route(question)
