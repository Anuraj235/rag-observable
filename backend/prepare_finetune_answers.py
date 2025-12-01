import json
from pathlib import Path
import re

# Update these to your actual paths
INPUT_PATH = Path("./data/fine_tune_dataset.jsonl")          # original file
OUTPUT_PATH = Path("./data/fine_tune_dataset_clean.jsonl")   # cleaned file


def split_sentences(text: str):
    """
    Very simple sentence splitter: splits on '. ' / '? ' / '! '.
    Keeps the punctuation at the end of each sentence.
    """
    # Normalize newlines
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    # Split on punctuation followed by a space or newline
    parts = re.split(r'([.?!])\s+', text)
    sentences = []

    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if not chunk:
            continue

        punct = ""
        if i + 1 < len(parts):
            punct = parts[i + 1]

        s = (chunk + punct).strip()
        if s:
            sentences.append(s)

    return sentences


def rewrite_assistant_content(content: str) -> str:
    """
    Turn the assistant's answer into:

    summary sentence
    - bullet 1
    - bullet 2
    ...

    If the answer is just "I don't have enough information.", keep as-is.
    """

    stripped = content.strip()

    # Keep refusals exactly as they are
    if stripped.startswith("I don't have enough information"):
        return stripped

    # Drop anything after "Sources:" – we do not want source lists in training
    sources_idx = stripped.find("Sources:")
    if sources_idx != -1:
        main = stripped[:sources_idx].strip()
    else:
        main = stripped

    if not main:
        return stripped

    # Split into sentences
    sentences = split_sentences(main)
    if not sentences:
        return stripped

    # First sentence = summary
    summary = sentences[0].strip()

    # Remaining sentences -> bullets
    bullets = [s.strip() for s in sentences[1:] if s.strip()]

    lines = [summary]
    if bullets:
        lines.append("")  # blank line
        for s in bullets:
            lines.append(f"- {s}")

    return "\n".join(lines).strip()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    n_in, n_out = 0, 0
    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            obj = json.loads(line)

            messages = obj.get("messages", [])

            # Find the assistant message and rewrite its content
            for msg in messages:
                if msg.get("role") == "assistant":
                    original = msg.get("content", "")
                    msg["content"] = rewrite_assistant_content(original)

            # ❗ IMPORTANT: no "metadata" field, only "messages"
            clean_obj = {
                "messages": messages
            }

            fout.write(json.dumps(clean_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Done. Read {n_in} examples, wrote {n_out} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
