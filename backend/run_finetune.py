import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env (so we get OPENAI_API_KEY)
load_dotenv()

client = OpenAI()

# Change this if your file has a different name or path
TRAINING_FILE_PATH =  "./data/fine_tune_dataset_clean.jsonl"


def main():
    if not os.path.exists(TRAINING_FILE_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE_PATH}")

    print(f"Uploading training file: {TRAINING_FILE_PATH} ...")

    # 1) Upload the JSONL dataset
    with open(TRAINING_FILE_PATH, "rb") as f:
        upload = client.files.create(
            file=f,
            purpose="fine-tune",
        )

    print("? Uploaded file id:", upload.id)

    # 2) Start a fine-tuning job on gpt-4o-mini snapshot
    job = client.fine_tuning.jobs.create(
        training_file=upload.id,
        model="gpt-4o-mini-2024-07-18",  # snapshot that supports fine-tuning
        # Optional: uncomment to control training epochs, etc.
        # method={
        #     "type": "supervised",
        #     "supervised": {
        #         "hyperparameters": {
        #             "n_epochs": 3,
        #         }
        #     },
        # },
    )

    print("?? Started fine-tune job:", job.id)
    print("Status:", job.status)
    print("Full job object:")
    print(job)


if __name__ == "__main__":
    main()
