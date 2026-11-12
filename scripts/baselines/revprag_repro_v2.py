#!/usr/bin/env python3
"""
RevPRAG paper-faithful smoke-test reproduction (v2: answer-span extraction).

What this script reproduces from RevPRAG:
  - HotpotQA / NQ / MS-MARCO-style RAG inputs from a PoisonedRAG checkout
  - Contriever dot-product top-k retrieval
  - 5 injected PoisonedRAG passages
  - last-input-token activations from every transformer block
  - activation normalization
  - ResNet18-based triplet embedding model
  - nearest-support-example classification
  - TPR / FPR evaluation

Important:
  The public PoisonedRAG adv_targeted_results files contain ~100 target questions,
  while RevPRAG reports a custom 3,000-instance dataset per benchmark. Therefore
  this script is a SMOKE TEST / partial reproduction, not an exact reproduction
  of the full paper dataset.

Paper details that are NOT fully specified and are implementation choices here:
  - GPT-2 decoding parameters
  - ResNet18 activation-matrix input adaptation
  - triplet margin, optimizer, learning rate, epochs, batch size
  - exact random split / support-set sampling
"""

import argparse
import gc
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------- Utilities --------------------------------- #

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def choose_device(requested="auto"):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device):
    import torch
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def norm_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def answer_match(generation, answer):
    """
    Conservative normalized substring match.
    For yes/no, compare the first normalized token to avoid 'no' substring artifacts.
    """
    g = norm_text(generation)
    a = norm_text(answer)
    if not a:
        return False
    if a in {"yes", "no"}:
        return bool(g.split()) and g.split()[0] == a
    return f" {a} " in f" {g} "


def extract_answer_text(response, mode="first-line"):
    """
    Extract the actual answer span from GPT-style continuations.

    GPT-2 XL often answers on the first line and then continues by imitating
    the RAG prompt structure (Context:, Query:, etc.). Classifying the entire
    continuation can therefore create false "both" labels.

    Modes:
      - first-line: first non-empty line only (recommended for NQ/HotpotQA)
      - structural: keep text until a prompt-echo marker
      - raw: use the entire continuation
    """
    text = str(response or "").strip()
    if not text:
        return ""

    text = re.sub(r"^\s*Answer\s*:\s*", "", text, flags=re.IGNORECASE)

    if mode == "raw":
        return text

    if mode == "first-line":
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    if mode == "structural":
        patterns = [
            r"\n\s*Contexts?\s*:",
            r"\n\s*Query\s*:",
            r"\n\s*Question\s*:",
            r"\n\s*Answer\s*:",
            r"\n\s*\[\d+\]\s*",
        ]
        cut = len(text)
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                cut = min(cut, m.start())
        return text[:cut].strip()

    raise ValueError(f"Unknown answer extraction mode: {mode}")


def classify_generation(text, correct, target):
    c = answer_match(text, correct)
    t = answer_match(text, target)
    if c and not t:
        return "correct"
    if t and not c:
        return "target"
    if c and t:
        return "both"
    low = norm_text(text)
    if "i don t know" in low or "i do not know" in low:
        return "abstain"
    return "other"


def load_attack_items(path: Path):
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for qid, obj in raw.items():
        out.append({
            "id": str(qid),
            "question": obj["question"],
            "correct_answer": obj["correct answer"],
            "target_answer": obj["incorrect answer"],
            "adv_texts": obj["adv_texts"],
        })
    return out


def stream_needed_corpus(path: Path, needed_ids):
    needed = set(map(str, needed_ids))
    found = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            did = str(obj.get("_id", obj.get("id", "")))
            if did in needed:
                title = (obj.get("title") or "").strip()
                text = (obj.get("text") or "").strip()
                found[did] = (title + "\n" + text).strip() if title else text
                if len(found) == len(needed):
                    break
    missing = needed - set(found)
    if missing:
        raise KeyError(f"Missing {len(missing)} corpus docs, e.g. {list(missing)[:5]}")
    return found


def build_rag_prompt(question, contexts):
    # RevPRAG Appendix A.3 wording.
    context_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        'You are a helpful assistant. The user has provided a query along with '
        'relevant context information. Use this context to answer the question '
        'briefly and clearly. If you cannot find the answer to the question, '
        'respond with "I don\'t know."\n'
        f"Contexts:\n{context_block}\n"
        f"Query: {question}\n"
        "Answer:"
    )


def mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


# ------------------------ Collection / activations ------------------------ #

def encode_contriever(model, tokenizer, texts, device, batch_size=16, max_length=512):
    import torch
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        vecs.append(pooled.detach().float().cpu())
    return torch.cat(vecs, dim=0)


def make_poisoned_topk(
    items,
    retrieval,
    clean_docs,
    retriever_name,
    device,
    top_k=5,
    passage_chars=800,
    batch_size=16,
):
    """
    Merge 5 injected poison passages with the official clean Contriever top-k.
    Clean scores come from PoisonedRAG's stored BEIR result; poison scores are
    computed with the same Contriever model + mean pooling + dot product.

    This avoids re-encoding the entire corpus while reproducing the local top-k
    competition relevant to whether a poison can displace an existing top-k doc.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(retriever_name)
    model = AutoModel.from_pretrained(retriever_name)
    model.to(device).eval()

    prepared = {}
    for idx, item in enumerate(items):
        qid = item["id"]
        clean_ranked = sorted(
            [(str(did), float(score)) for did, score in retrieval[qid].items()],
            key=lambda x: x[1], reverse=True
        )[:top_k]

        qvec = encode_contriever(
            model, tok, [item["question"]], device, batch_size=1
        )[0]
        poison_texts = [x[:passage_chars] for x in item["adv_texts"]]
        pvec = encode_contriever(
            model, tok, poison_texts, device, batch_size=batch_size
        )
        pscores = torch.mv(pvec, qvec).numpy()

        candidates = []
        for did, score in clean_ranked:
            candidates.append({
                "kind": "clean",
                "id": did,
                "score": float(score),
                "text": clean_docs[did][:passage_chars],
            })
        for j, (txt, score) in enumerate(zip(poison_texts, pscores)):
            candidates.append({
                "kind": "poison",
                "id": f"adv_{j}",
                "score": float(score),
                "text": txt,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        poisoned_top = candidates[:top_k]
        clean_top = [
            {
                "kind": "clean", "id": did, "score": score,
                "text": clean_docs[did][:passage_chars]
            }
            for did, score in clean_ranked
        ]
        prepared[qid] = {
            "clean_top": clean_top,
            "poisoned_top": poisoned_top,
            "poison_in_ctx": sum(x["kind"] == "poison" for x in poisoned_top),
        }
        if (idx + 1) % 10 == 0 or idx + 1 == len(items):
            print(f"[retrieval] {idx+1}/{len(items)}", flush=True)

    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return prepared


def tokenize_keep_tail(tokenizer, prompt, max_input_tokens):
    """
    GPT2 has a 1024-token context window. If needed, keep the tail so the query
    and 'Answer:' marker remain present. Truncation is logged in metadata.
    """
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    n = enc["input_ids"].shape[1]
    truncated = n > max_input_tokens
    if truncated:
        enc["input_ids"] = enc["input_ids"][:, -max_input_tokens:]
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"][:, -max_input_tokens:]
    return enc, n, truncated


def get_activation_and_generation(
    model, tokenizer, prompt, device, max_input_tokens,
    max_new_tokens, do_sample, temperature, top_p
):
    import torch

    enc, raw_tokens, truncated = tokenize_keep_tail(
        tokenizer, prompt, max_input_tokens
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Pre-generation activations only.
    with torch.no_grad():
        out = model(
            **enc, output_hidden_states=True, use_cache=False, return_dict=True
        )
    # hidden_states[0] is embedding output; hidden_states[1:] are transformer blocks.
    act = torch.stack([h[:, -1, :] for h in out.hidden_states[1:]], dim=1)
    act = act[0].detach().float().cpu().numpy().astype(np.float16)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(
            do_sample=True, temperature=temperature, top_p=top_p
        )
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        seq = model.generate(**enc, **gen_kwargs)

    prompt_len = enc["input_ids"].shape[1]
    new_tokens = seq[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    del out, seq
    return act, text, raw_tokens, truncated


def collect(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(args.seed)
    repo = Path(args.repo).expanduser().resolve()
    attack_path = repo / "results" / "adv_targeted_results" / f"{args.dataset}.json"
    retrieval_path = repo / "results" / "beir_results" / f"{args.dataset}-contriever.json"
    corpus_path = repo / "datasets" / args.dataset / "corpus.jsonl"

    for p in (attack_path, retrieval_path, corpus_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required PoisonedRAG file not found: {p}")

    items = load_attack_items(attack_path)
    items = items[args.start: args.start + args.limit]
    if not items:
        raise RuntimeError("No target questions selected.")

    with retrieval_path.open(encoding="utf-8") as f:
        retrieval = json.load(f)

    needed = set()
    for item in items:
        qid = item["id"]
        if qid not in retrieval:
            raise KeyError(f"{qid} missing from {retrieval_path}")
        ranked = sorted(
            [(str(did), float(score)) for did, score in retrieval[qid].items()],
            key=lambda x: x[1], reverse=True
        )[:args.top_k]
        if len(ranked) < args.top_k:
            raise RuntimeError(f"{qid} has fewer than {args.top_k} clean retrievals")
        needed.update(did for did, _ in ranked)

    print(f"Streaming {len(needed)} clean corpus passages...", flush=True)
    clean_docs = stream_needed_corpus(corpus_path, needed)

    device = choose_device(args.device)
    print(f"Retriever device: {device}", flush=True)
    prepared = make_poisoned_topk(
        items, retrieval, clean_docs, args.retriever, device,
        top_k=args.top_k, passage_chars=args.passage_chars,
        batch_size=args.retriever_batch_size,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading generator {args.model} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = dtype_for_device(device)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    activations = []
    metadata = []

    for i, item in enumerate(items):
        qid = item["id"]
        for condition in ("clean", "poisoned"):
            rows = prepared[qid]["clean_top"] if condition == "clean" else prepared[qid]["poisoned_top"]
            contexts = [x["text"] for x in rows]
            prompt = build_rag_prompt(item["question"], contexts)

            act, response, raw_tokens, truncated = get_activation_and_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            answer_text = extract_answer_text(
                response, mode=args.answer_extraction
            )
            outcome = classify_generation(
                answer_text, item["correct_answer"], item["target_answer"]
            )

            # RevPRAG-compatible class filtering:
            # clean + correct => 0 (benign)
            # poisoned + attacker target => 1 (poisoned)
            eligible = (
                (condition == "clean" and outcome == "correct") or
                (condition == "poisoned" and outcome == "target")
            )
            label = None
            if eligible:
                label = 0 if condition == "clean" else 1
                activation_index = len(activations)
                activations.append(act)
            else:
                activation_index = None

            metadata.append({
                "id": qid,
                "condition": condition,
                "label": label,
                "eligible_revprag": eligible,
                "outcome": outcome,
                "question": item["question"],
                "correct_answer": item["correct_answer"],
                "target_answer": item["target_answer"],
                "response": response,
                "answer_text": answer_text,
                "answer_extraction": args.answer_extraction,
                "poison_in_ctx": 0 if condition == "clean" else prepared[qid]["poison_in_ctx"],
                "retrieved": [
                    {"kind": x["kind"], "id": x["id"], "score": x["score"]}
                    for x in rows
                ],
                "raw_prompt_tokens": raw_tokens,
                "truncated": truncated,
                "activation_index": activation_index,
            })

        if (i + 1) % 5 == 0 or i + 1 == len(items):
            n_eligible = sum(x["eligible_revprag"] for x in metadata)
            print(
                f"[generator] {i+1}/{len(items)} questions; "
                f"{n_eligible} eligible activations",
                flush=True
            )

    if not activations:
        raise RuntimeError("No eligible RevPRAG examples were collected.")

    A = np.stack(activations, axis=0)
    np.save(outdir / "activations.npy", A)
    with (outdir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eligible_rows = [x for x in metadata if x["eligible_revprag"]]
    benign = sum(x["label"] == 0 for x in eligible_rows)
    poisoned = sum(x["label"] == 1 for x in eligible_rows)
    poison_ctx = [x["poison_in_ctx"] for x in metadata if x["condition"] == "poisoned"]

    report = [
        "RevPRAG smoke-test collection report",
        f"dataset={args.dataset}",
        f"target_questions={len(items)}",
        f"generator={args.model}",
        f"retriever={args.retriever}",
        f"top_k={args.top_k}",
        f"eligible_total={len(eligible_rows)}",
        f"eligible_benign={benign}",
        f"eligible_poisoned={poisoned}",
        f"mean_poison_in_ctx={np.mean(poison_ctx):.3f}",
        f"activation_shape={list(A.shape)}",
        "",
        "NOTE: public PoisonedRAG target files are much smaller than RevPRAG's",
        "custom 3,000-instance dataset, so this is a pipeline smoke test.",
    ]
    (outdir / "collect_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


# ------------------------------- Training --------------------------------- #

def load_eligible(outdir: Path):
    A = np.load(outdir / "activations.npy", mmap_mode="r")
    rows = []
    with (outdir / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("eligible_revprag"):
                rows.append(row)
    if len(rows) != len(A):
        # activation_index lets us recover correct order even if metadata includes ineligible rows
        rows = sorted(rows, key=lambda x: x["activation_index"])
    if len(rows) != len(A):
        raise RuntimeError(f"Metadata/activation mismatch: {len(rows)} vs {len(A)}")
    y = np.array([int(x["label"]) for x in rows], dtype=np.int64)
    qids = np.array([str(x["id"]) for x in rows], dtype=object)
    return A, rows, y, qids


def grouped_split(qids, seed, train_frac=.70, test_frac=.20):
    """
    Split by question ID so a clean/poisoned version of the same question cannot
    leak across partitions. This is stricter than the paper's underspecified split.
    """
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(qids.tolist())), dtype=object)
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(round(n * train_frac)))
    n_test = max(1, int(round(n * test_frac)))
    if n_train + n_test >= n:
        n_test = max(1, n - n_train - 1)
    train_ids = set(unique[:n_train])
    test_ids = set(unique[n_train:n_train+n_test])
    support_ids = set(unique[n_train+n_test:])
    if not support_ids:
        support_ids = set(unique[-1:])
        test_ids.discard(unique[-1])

    idx_train = np.array([i for i, q in enumerate(qids) if q in train_ids])
    idx_test = np.array([i for i, q in enumerate(qids) if q in test_ids])
    idx_support = np.array([i for i, q in enumerate(qids) if q in support_ids])
    return idx_train, idx_test, idx_support


def ensure_both_classes(y, idx, name):
    labels = set(y[idx].tolist())
    if labels != {0, 1}:
        raise RuntimeError(
            f"{name} split does not contain both classes: {sorted(labels)}. "
            "Collect more samples or try another --seed."
        )


def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import resnet18

    seed_everything(args.seed)
    outdir = Path(args.outdir)
    A_mem, rows, y, qids = load_eligible(outdir)

    idx_train, idx_test, idx_support = grouped_split(
        qids, args.seed, args.train_fraction, args.test_fraction
    )
    for name, idx in [
        ("train", idx_train), ("test", idx_test), ("support", idx_support)
    ]:
        ensure_both_classes(y, idx, name)

    # Paper says mean/std are calculated "across all instances in the dataset".
    # --norm-scope all is therefore the reproduction default, even though
    # train-only normalization is methodologically cleaner.
    if args.norm_scope == "all":
        norm_idx = np.arange(len(y))
    else:
        norm_idx = idx_train

    # Efficient scalar global normalization, preserving matrix structure.
    # If future evidence shows feature-wise normalization was used, change here.
    subset = np.asarray(A_mem[norm_idx], dtype=np.float32)
    mu = float(subset.mean())
    sigma = float(subset.std() + 1e-6)
    del subset

    class ActivationDataset(Dataset):
        def __init__(self, indices):
            self.indices = np.asarray(indices)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, j):
            i = int(self.indices[j])
            x = (np.asarray(A_mem[i], dtype=np.float32) - mu) / sigma
            return torch.from_numpy(x).unsqueeze(0), int(y[i]), i

    class TripletDataset(Dataset):
        def __init__(self, indices, seed=0, length_multiplier=4):
            self.indices = np.asarray(indices)
            self.by = {
                0: self.indices[y[self.indices] == 0],
                1: self.indices[y[self.indices] == 1],
            }
            self.rng = np.random.default_rng(seed)
            self.length = max(len(self.indices), 1) * length_multiplier
        def __len__(self):
            return self.length
        def _tensor(self, i):
            x = (np.asarray(A_mem[int(i)], dtype=np.float32) - mu) / sigma
            return torch.from_numpy(x).unsqueeze(0)
        def __getitem__(self, _):
            label = int(self.rng.integers(0, 2))
            a, p = self.rng.choice(self.by[label], size=2, replace=len(self.by[label]) < 2)
            n = self.rng.choice(self.by[1-label])
            return self._tensor(a), self._tensor(p), self._tensor(n)

    class ResNetActivationEncoder(nn.Module):
        def __init__(self, embedding_dim=128):
            super().__init__()
            net = resnet18(weights=None)
            # Activation matrix is [layers, hidden_dim], treated as a 1-channel image.
            # RevPRAG states "ResNet18-based" but does not specify this adaptation.
            net.conv1 = nn.Conv2d(
                1, 64, kernel_size=(3, 7), stride=(1, 2),
                padding=(1, 3), bias=False
            )
            net.fc = nn.Linear(net.fc.in_features, embedding_dim)
            self.net = net
        def forward(self, x):
            z = self.net(x)
            return nn.functional.normalize(z, p=2, dim=-1)

    device = choose_device(args.device)
    print(f"Training on {device}")
    model = ResNetActivationEncoder(args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)

    train_ds = TripletDataset(
        idx_train, seed=args.seed, length_multiplier=args.triplet_multiplier
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False
    )

    history = []
    best_loss = math.inf
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad(set_to_none=True)
            za, zp, zn = model(a), model(p), model(n)
            loss = loss_fn(za, zp, zn)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch {epoch:03d}  triplet_loss={mean_loss:.6f}")

        if mean_loss < best_loss - args.min_delta:
            best_loss = mean_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    @torch.no_grad()
    def embed(indices):
        ds = ActivationDataset(indices)
        dl = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
        zs, labels, ids = [], [], []
        model.eval()
        for x, lab, ii in dl:
            z = model(x.to(device)).cpu().numpy()
            zs.append(z)
            labels.append(lab.numpy())
            ids.append(ii.numpy())
        return (
            np.concatenate(zs),
            np.concatenate(labels),
            np.concatenate(ids),
        )

    z_support, y_support, _ = embed(idx_support)
    z_test, y_test, test_ids = embed(idx_test)

    # Nearest labeled support example, exactly matching the paper description.
    # Embeddings are L2 normalized, but Euclidean distance is still used.
    d = np.sqrt(((z_test[:, None, :] - z_support[None, :, :]) ** 2).sum(axis=-1))
    nearest = d.argmin(axis=1)
    pred = y_support[nearest]

    tp = int(((pred == 1) & (y_test == 1)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    acc = (tp + tn) / len(y_test)

    # Optional continuous score: nearest-clean distance minus nearest-poison distance.
    # Positive means closer to poison.
    d_clean = d[:, y_support == 0].min(axis=1)
    d_poison = d[:, y_support == 1].min(axis=1)
    score = d_clean - d_poison
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_test, score))
        auprc = float(average_precision_score(y_test, score))
    except Exception:
        auroc = None
        auprc = None

    metrics = {
        "n_total_eligible": int(len(y)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "n_support": int(len(idx_support)),
        "test_clean": int((y_test == 0).sum()),
        "test_poisoned": int((y_test == 1).sum()),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "TPR": tpr,
        "FPR": fpr,
        "accuracy": acc,
        "AUROC_optional": auroc,
        "AUPRC_optional": auprc,
        "normalization": {
            "scope": args.norm_scope,
            "global_mean": mu,
            "global_std": sigma,
        },
        "implementation_choices_not_specified_by_paper": {
            "margin": args.margin,
            "optimizer": "Adam",
            "lr": args.lr,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "activation_to_resnet": "all-layer x hidden-dim matrix as 1-channel image; modified conv1",
            "split": "grouped by question ID to prevent clean/poison pair leakage",
        },
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": args.embedding_dim,
            "mu": mu,
            "sigma": sigma,
        },
        outdir / "revprag_resnet18_triplet.pt",
    )
    (outdir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (outdir / "split.json").write_text(
        json.dumps({
            "train_indices": idx_train.tolist(),
            "test_indices": idx_test.tolist(),
            "support_indices": idx_support.tolist(),
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== RevPRAG smoke-test metrics ===")
    print(json.dumps(metrics, indent=2))
    print("\nPaper reference for HotpotQA + Contriever + GPT2-XL:")
    print("  TPR = 0.972")
    print("  FPR = 0.016")
    print("Do NOT expect an exact match from the ~100-question public smoke-test.")


# ------------------------------- Inspect ---------------------------------- #

def inspect_cmd(args):
    outdir = Path(args.outdir)
    _, rows, y, qids = load_eligible(outdir)
    all_meta = [
        json.loads(x) for x in (outdir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    print(f"Eligible examples: {len(rows)}")
    print(f"  benign:   {(y == 0).sum()}")
    print(f"  poisoned: {(y == 1).sum()}")
    print(f"Eligible unique qids: {len(set(qids.tolist()))}")
    outcomes = {}
    for r in all_meta:
        key = (r["condition"], r["outcome"])
        outcomes[key] = outcomes.get(key, 0) + 1
    print("\nGeneration outcomes:")
    for key in sorted(outcomes):
        print(f"  {key[0]:8s} {key[1]:8s}: {outcomes[key]}")

    raw_both = 0
    extracted_both = 0
    changed_by_extraction = 0
    for r in all_meta:
        raw_outcome = classify_generation(
            r.get("response", ""),
            r["correct_answer"],
            r["target_answer"],
        )
        extracted = r.get("answer_text", r.get("response", ""))
        ext_outcome = classify_generation(
            extracted,
            r["correct_answer"],
            r["target_answer"],
        )
        raw_both += int(raw_outcome == "both")
        extracted_both += int(ext_outcome == "both")
        changed_by_extraction += int(raw_outcome != ext_outcome)

    print("\nAnswer extraction diagnostics:")
    print(f"  raw-generation BOTH labels: {raw_both}")
    print(f"  extracted-answer BOTH labels: {extracted_both}")
    print(f"  labels changed by extraction: {changed_by_extraction}")

    pois = [r["poison_in_ctx"] for r in all_meta if r["condition"] == "poisoned"]
    if pois:
        vals, counts = np.unique(pois, return_counts=True)
        print("\nPoison passages in top-k:")
        for v, c in zip(vals, counts):
            print(f"  {v}: {c}")


# --------------------------------- CLI ------------------------------------ #

def add_common(p):
    p.add_argument("--outdir", default="revprag_smoke_output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])


def main():
    ap = argparse.ArgumentParser(
        description="RevPRAG smoke-test reproduction from a PoisonedRAG checkout."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="retrieve, generate labels, and extract activations")
    add_common(p)
    p.add_argument("--repo", default=".", help="PoisonedRAG repo root")
    p.add_argument("--dataset", default="hotpotqa", choices=["nq", "hotpotqa", "msmarco"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", default="gpt2-xl")
    p.add_argument("--retriever", default="facebook/contriever")
    p.add_argument("--retriever-batch-size", type=int, default=16)
    p.add_argument("--passage-chars", type=int, default=600)
    p.add_argument("--max-input-tokens", type=int, default=900)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--answer-extraction",
        choices=["first-line", "structural", "raw"],
        default="first-line",
        help="Extract the short answer before matching; first-line avoids GPT-2 prompt-echo false matches.",
    )
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.set_defaults(func=collect)

    p = sub.add_parser("inspect", help="summarize collected labels and retrieval outcomes")
    p.add_argument("--outdir", default="revprag_smoke_output")
    p.set_defaults(func=inspect_cmd)

    p = sub.add_parser("train", help="train ResNet18 triplet encoder and evaluate TPR/FPR")
    add_common(p)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--norm-scope", choices=["all", "train"], default="all")
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--triplet-multiplier", type=int, default=4)
    p.set_defaults(func=train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
