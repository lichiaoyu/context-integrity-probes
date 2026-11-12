# Context Integrity Probes (CIP) — repository starter

Research scripts for controlled RAG-context poisoning detection, hidden-state probes, attention analysis, and paraphrase robustness. **This is a curated starter, not yet a verified end-to-end reproduction.**

## Scope and provenance

- `scripts/`: selected files copied **unchanged** from the AICR script backup. File grouping is organizational; not all scripts are independently executable from these subdirectories.
- `jobs/`: AICR SLURM batch scripts, copied unchanged. Their relative paths and cluster-specific paths may need updating.
- `archive/aicr/`: all AICR Python files for reference, **excluded from Git by `.gitignore`**.
- Not included: original PoisonedRAG upstream repository, Mac-local Semantic Entropy/RevPRAG scripts, Mac-local `ceg_reconstruct_matched.py`, raw corpora, labels, activations, OOF scores, and paraphrase JSONL. Add reviewed original work and explicit provenance separately.

## Experimental stages

1. **Construct matched contexts** — MS MARCO controlled replacement, 449 question pairs / 898 clean-poison contexts in the reported run. **The original construction script was not identified in this AICR backup.** The separately held `ceg_reconstruct_matched.py` is a reconstruction and must not be represented as the original pipeline. Controlled replacement is **not real retrieval**.
2. **Extract hidden states** — `scripts/extraction/ceg_extract_tbg.py`; Qwen2.5-1.5B-Instruct by default, passage cap 650 characters, model input cap 2048 tokens. Check actual arguments and metadata before reproduction.
3. **Train/evaluate probes** — `scripts/probes/ceg_fourprobe_eval.py`, which imports `camp_train_nq_probes.Probe`. Validate labels and question-grouped splits.
4. **Paraphrase** — `scripts/robustness/ceg_paraphrase_pilot.py`; Qwen2.5-7B-Instruct default.
5. **Cross-version evaluation** — `ceg_crossversion_449.py`, `ceg_crossversion_4way.py`; evaluate O→O, O→Q, Q→Q, Q→O.
6. **Low-FPR and OOF** — `ceg_lowfpr_eval.py` imports `load_dataset` from `ceg_crossversion_449.py`; writes out-of-fold scores. Repeated seeds are **not independent question examples**.
7. **Attention** — attention-based **presence classification** and raw-attention **passage localization** are different tasks. A teammate reported 73% poisoned-passage localization after paraphrasing; this figure has not been independently verified here.

## Running scripts

These scripts were authored in a flat working directory and some import each other by module name. For initial reproducibility, run from a working directory containing the needed scripts or set `PYTHONPATH` to include their respective folders, and pass explicit data paths. Check SLURM scripts before running on a different cluster. Do not run training from a login node.

## Before publishing

- Review every script and dependency, license and third-party attribution.
- Check for hard-coded personal paths, secrets, tokens, and confidential dataset contents.
- Verify source files and provenance for the 449 matched pairs.
- Document environment versions, exact commands, seeds, data checksums, and evaluation splits.
- Review `git status` and `git diff --cached` before pushing.

This folder is **not** initialized as a Git repository; create a new independent repo when ready.
