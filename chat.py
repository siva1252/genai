import os
import sys

from dotenv import load_dotenv

from llm import generate_answer
from router import choose_mode, get_current_time_context
from search import is_doc_relevant, search_chunks
from web_search import web_search

load_dotenv()


def get_doc_context(question):
    context, distance = search_chunks(question, n_results=5)
    return context, distance


def chat():
    if not os.getenv("SARVAM_API_KEY"):
        print("Error: SARVAM_API_KEY not found in .env file.")
        return

    print("Smart Q&A (type 'exit' or 'quit' to stop)")
    print("  Just ask normally — LLM auto-routes DOC / WEB / TIME")
    print("  Optional force:  web: question   or   doc: question\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            lower = user_input.lower()
            forced = False

            if lower.startswith("web:"):
                question = user_input[4:].strip()
                mode = "WEB"
                forced = True
            elif lower.startswith("doc:"):
                question = user_input[4:].strip()
                mode = "DOC"
                forced = True
            else:
                question = user_input
                print("(Routing...)")
                mode = choose_mode(question)

            if not question:
                print("\nPlease type a question.\n")
                continue

            # DOC path with relevance check (no keyword hints)
            if mode == "DOC":
                print("(Using DOC...)")
                context, distance = get_doc_context(question)

                # If PDF is not similar enough, and not forced, use WEB
                if (
                    not forced
                    and os.getenv("TAVILY_API_KEY")
                    and not is_doc_relevant(distance)
                ):
                    print("(DOC not relevant enough → switching to WEB...)")
                    mode = "WEB"
                    context = web_search(question)
                elif not context:
                    print("\n[DOC] No context found in the document.\n")
                    continue

            elif mode == "TIME":
                print("(Using TIME...)")
                context = get_current_time_context()

            else:
                print("(Using WEB...)")
                if not os.getenv("TAVILY_API_KEY"):
                    raise ValueError("TAVILY_API_KEY not found in .env file.")
                context = web_search(question)

            if not context:
                print(f"\n[{mode}] No context found. Try rephrasing.\n")
                continue

            answer = generate_answer(question, context)
            print(f"\n[{mode}] Answer: {answer}\n")
        except ValueError as e:
            print(f"\nError: {e}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    chat()
