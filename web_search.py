import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def web_search(question, max_results=5):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in .env file.")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=question,
        max_results=max_results,
        include_answer=True,
        search_depth="basic",
    )

    parts = []

    # Tavily short answer (if available)
    answer = response.get("answer")
    if answer:
        parts.append(f"Search summary: {answer}")

    results = response.get("results") or []
    for i, item in enumerate(results, start=1):
        title = item.get("title") or ""
        url = item.get("url") or ""
        content = item.get("content") or ""
        if not (title or content):
            continue
        parts.append(
            f"Source {i}: {title}\nURL: {url}\nContent: {content}"
        )

    return "\n\n".join(parts)


if __name__ == "__main__":
    question = input("Web search question: ").strip()
    if question:
        print(web_search(question))
    else:
        print("No question entered.")
