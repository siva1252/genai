from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER")
)

messages = [
    {
        "role": "system",
        "content": "You are a Python GenAI interviewer."
    }
]

while True:

    user_input = input("Enter your message: ")

    if user_input.lower() == "bye":
        print("Exiting the chat...")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="liquid/lfm-2.5-2.6b:free",
        messages=messages
    )

    assistant_response = response.choices[0].message.content

    messages.append({
        "role": "assistant",
        "content": assistant_response
    })

    print("Assistant:", assistant_response)




# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# Jarvis = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPEN_ROUTER")
    
# )


# responce = Jarvis.responses.create(
#     model="liquid/lfm-2.5-2.6b:free",
#     input=("entry the message")
# )

# print(Jarvis.responses.output_text)