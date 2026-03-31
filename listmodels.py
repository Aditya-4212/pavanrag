import os
from google import genai

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise Exception("Please set the GEMINI_API_KEY environment variable")

API_KEY2 = os.environ.get("CHATTING_GEMINI_API_KEY")
if not API_KEY2:
    raise Exception("Please set the CHATTING_GEMINI_API_KEY environment variable")


def list_models(API_KEY):
    if not API_KEY:
        raise ValueError("API Key is required")

    client = genai.Client(api_key=API_KEY)

    models = client.models.list()

    print("Available Models:\n")
    for model in models:
        print(model.name)

# Example usage
API_KEY = API_KEY
list_models(API_KEY)