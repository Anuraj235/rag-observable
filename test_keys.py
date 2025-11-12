from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os, requests

# Find .env starting from the current working dir; fall back to file's folder
env_path = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parent / ".env")
load_dotenv(env_path, override=True)
print("Loaded .env from:", env_path)

openai_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HUGGINGFACE_API_KEY")

print("\n🔍 Testing API Keys...\n")

# --- Test OpenAI key ---
if openai_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        client.models.list()
        print("✅ OpenAI key works! Found models.")
    except Exception as e:
        print("❌ OpenAI key error:", e)
else:
    print("⚠️ OpenAI key missing!")

# --- Test Hugging Face key ---
if hf_key:
    try:
        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {hf_key}"}
        )
        if r.status_code == 200:
            print("✅ Hugging Face key works! User:", r.json().get("name"))
        else:
            print("❌ Hugging Face key error:", r.status_code, r.text[:200])
    except Exception as e:
        print("❌ Hugging Face test failed:", e)
else:
    print("⚠️ Hugging Face key missing!")
