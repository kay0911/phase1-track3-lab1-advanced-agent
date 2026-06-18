import json
import random
import argparse
from pathlib import Path

try:
    from src.reflexion_lab.schemas import QAExample
except ImportError:
    QAExample = None

def main():
    parser = argparse.ArgumentParser(description="Sample random examples from HotpotQA dataset.")
    parser.add_argument("--input", default="data/hotpot_dev_distractor_v1.json", help="Path to input hotpot JSON file")
    parser.add_argument("--output", default="data/hotpot_100.json", help="Path to output JSON file")
    parser.add_argument("--count", type=int, default=100, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total items in source: {len(data)}")
    if len(data) < args.count:
        print(f"Warning: requested count {args.count} is larger than dataset size {len(data)}. Sampling all.")
        sampled = data
    else:
        random.seed(args.seed)
        sampled = random.sample(data, args.count)

    processed = []
    for item in sampled:
        # Map raw fields to QAExample format
        # HotpotQA raw context is [[title, [sentence1, sentence2, ...]], ...]
        context_chunks = []
        for title, sentences in item.get("context", []):
            joined_text = " ".join(s.strip() for s in sentences)
            context_chunks.append({
                "title": title,
                "text": joined_text
            })

        qa_item = {
            "qid": item.get("_id"),
            "difficulty": item.get("level", "medium"),
            "question": item.get("question"),
            "gold_answer": item.get("answer"),
            "context": context_chunks
        }

        # Validate if QAExample is available
        if QAExample is not None:
            try:
                QAExample.model_validate(qa_item)
            except Exception as e:
                print(f"Validation error for item {qa_item.get('qid')}: {e}")
                continue

        processed.append(qa_item)

    print(f"Saving {len(processed)} validated items to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print("Sampling completed successfully!")

if __name__ == "__main__":
    main()
