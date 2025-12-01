from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI()

MODEL = os.getenv("FINE_TUNED_MODEL")

def main():
    print("Using model:", MODEL)

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": "You are a careful instructor for a Retrieval-Augmented Generation (RAG) system."
            },
            {
                "role": "user",
                "content": (
                    "Question: what is data science?\n\n"
                    "Context items:\n"
                    "[1] Data science is an interdisciplinary field that uses scientific methods "
                    "and processes to extract insights and knowledge from data.\n"
                )
            }
        ]
    )

    print(resp.output[0].content[0].text)

if __name__ == "__main__":
    main()
