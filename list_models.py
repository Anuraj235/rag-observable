from openai import OpenAI
from dotenv import load_dotenv
import os, json

# Load your .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
models = client.models.list()

print(json.dumps([m.id for m in models.data], indent=2))
