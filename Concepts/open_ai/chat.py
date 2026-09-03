
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER")
)


response = client.responses.create(
    model="liquid/lfm-2.5-2.6b:free",
    input=input("entry the input message:")

)

print(response.output_text)