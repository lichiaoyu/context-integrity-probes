import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_paraphrase(model, tok, text, question, role, max_new_tokens):
    instruction = f"""Rewrite the following retrieved passage in clear, natural English.

Rules:
- Preserve every factual claim, number, entity, and answer-relevant detail.
- Do not correct, fact-check, or remove false claims.
- Do not introduce new facts or explanations.
- Preserve the original answer to the question, if the passage provides one.
- Use ordinary prose with consistent punctuation and spacing.
- Avoid unnecessary introductory phrases.
- Keep approximately the same length in words.
- Output only the rewritten passage.

Question (context only): {question}

Passage:
{text}"""

    messages = [{"role": "user", "content": instruction}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )

    result = tok.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    result = re.sub(r"^(Rewritten passage|Paraphrase):\s*", "", result, flags=re.I)
    if not result:
        raise RuntimeError(f"Empty paraphrase for {role}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(dict)
    with open(args.input) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                key = str(row["id"])
                condition = row["condition"]
                if condition in groups[key]:
                    raise ValueError(f"Duplicate: {key}, {condition}")
                groups[key][condition] = row

    valid = {}
    for key, pair in groups.items():
        if set(pair) != {"clean", "poison_1"}:
            continue
        clean, poison = pair["clean"], pair["poison_1"]
        if clean["question"] != poison["question"]:
            raise ValueError(f"Question mismatch: {key}")
        if clean["clean_answer"] != poison["clean_answer"]:
            raise ValueError(f"Clean answer mismatch: {key}")
        if clean["desired_answer"] != poison["desired_answer"]:
            raise ValueError(f"Desired answer mismatch: {key}")
        if len(clean["passages"]) != 5 or len(poison["passages"]) != 5:
            raise ValueError(f"Expected 5 passages: {key}")

        diff = [
            i for i in range(5)
            if clean["passages"][i] != poison["passages"][i]
        ]
        if diff != [poison["poison_position"] - 1]:
            raise ValueError(f"Poison position mismatch: {key}, {diff}")
        valid[key] = pair

    selected = sorted(valid)[:args.pairs]
    if len(selected) != args.pairs:
        raise ValueError("Not enough valid pairs")

    print(f"Selected {len(selected)} pairs", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    checkpoint = outdir / "completed_pairs.jsonl"
    completed = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record["id"] in completed:
                    raise ValueError("Duplicate checkpoint")
                completed[record["id"]] = record

    for index, key in enumerate(selected, 1):
        if key in completed:
            continue

        clean = valid[key]["clean"]
        poison = valid[key]["poison_1"]
        pos = poison["poison_position"] - 1

        clean_passages = []
        poison_passages = []

        for i in range(5):
            if i == pos:
                c = generate_paraphrase(
                    model, tok, clean["passages"][i],
                    clean["question"], "clean", args.max_new_tokens
                )
                p = generate_paraphrase(
                    model, tok, poison["passages"][i],
                    poison["question"], "poison", args.max_new_tokens
                )
            else:
                c = generate_paraphrase(
                    model, tok, clean["passages"][i],
                    clean["question"], "shared", args.max_new_tokens
                )
                p = c

            clean_passages.append(c)
            poison_passages.append(p)

        clean_new = dict(clean, passages=clean_passages)
        poison_new = dict(poison, passages=poison_passages)

        record = {
            "id": key,
            "clean": clean_new,
            "poison_1": poison_new,
        }

        with checkpoint.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed[key] = record
        print(f"Completed {index}/{len(selected)} pairs", flush=True)

    final_path = outdir / "paraphrased.jsonl"
    with final_path.open("w") as f:
        for key in selected:
            for condition in ("clean", "poison_1"):
                f.write(json.dumps(
                    completed[key][condition], ensure_ascii=False
                ) + "\n")

    print("Saved:", final_path, flush=True)


if __name__ == "__main__":
    main()
