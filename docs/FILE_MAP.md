# File map

The AICR archive contained 61 files (Python and SLURM). Curated scripts are unchanged copies.

**Core dependencies:** `ceg_fourprobe_eval.py` imports `camp_train_nq_probes.Probe`; `ceg_lowfpr_eval.py` and `ceg_crossversion_4way.py` import `ceg_crossversion_449.load_dataset`. These dependencies must remain importable when reorganizing.

**Missing from this backup:** MS MARCO 449-pair source construction script, NQ surface-match reconstruction script, original PoisonedRAG dependencies, data, and result outputs.

**Historical/specialized scripts:** remain in `archive/aicr` for review, not automatically included in version control.
