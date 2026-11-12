import json
import re
import string
from pathlib import Path
from statistics import mean

BASE = Path("camp_ceg_paraphrase_pilot")
FILES = {
    "Original": BASE / "pilot_100_original.jsonl",
    "Qwen": BASE / "smoke_10/paraphrased.jsonl",
    "Llama": BASE / "llama_smoke_10/paraphrased.jsonl",
    "Controlled Qwen": BASE / "qwen_controlled_10/paraphrased.jsonl",
}

def load(path):
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    data = {}
    for r in rows:
        key = (r["id"], r["condition"])
        if key in data:
            raise ValueError(f"Duplicate row: {key}")
        data[key] = r
    return data

def features(text):
    text = text[:650]
    words = re.findall(r"\b\w+\b", text)
    return {
        "chars": len(text),
        "words": len(words),
        "punct": sum(c in string.punctuation for c in text),
        "uppercase": sum(c.isupper() for c in text),
        "spaced_punct": len(re.findall(r"\s[.,!?;:]", text)),
    }

data = {name: load(path) for name, path in FILES.items()}

# Compare exactly the same matched questions across versions.
ids = set.intersection(*[
    {id_ for id_, condition in rows if condition == "clean"}
    & {id_ for id_, condition in rows if condition == "poison_1"}
    for rows in data.values()
])

print("Common matched pairs:", len(ids))
if not ids:
    raise SystemExit("No common matched pairs found")

for name, rows in data.items():
    clean_features = []
    poison_features = []
    shortest_count = 0
    shortest_total = 0
    changed_positions = []
    bad_pairs = 0

    for id_ in sorted(ids):
        clean = rows[(id_, "clean")]
        poison = rows[(id_, "poison_1")]
        cp = clean["passages"]
        pp = poison["passages"]

        if len(cp) != 5 or len(pp) != 5:
            bad_pairs += 1
            continue

        # Identify actual differing passage positions.
        changed = [i for i in range(5) if cp[i] != pp[i]]
        changed_positions.append(len(changed))

        if len(changed) != 1:
            bad_pairs += 1
            continue

        pos = changed[0]
        clean_features.append(features(cp[pos]))
        poison_features.append(features(pp[pos]))

        lengths = [len(p[:650].split()) for p in pp]
        if lengths[pos] == min(lengths):
            shortest_count += 1
        shortest_total += 1

    print(f"\n=== {name} ===")
    print("Valid one-changed-passage pairs:", len(clean_features))
    print("Invalid pairs:", bad_pairs)
    print("Changed-position counts:", {
        n: changed_positions.count(n) for n in sorted(set(changed_positions))
    })

    if not clean_features:
        continue

    print("Poison is shortest by words:",
          f"{shortest_count}/{shortest_total}",
          f"({shortest_count/shortest_total:.1%})")

    for feature in ["chars", "words", "punct", "uppercase", "spaced_punct"]:
        c = mean(f[feature] for f in clean_features)
        p = mean(f[feature] for f in poison_features)
        print(f"{feature:14s} clean={c:8.2f} poison={p:8.2f} delta={p-c:+8.2f}")
