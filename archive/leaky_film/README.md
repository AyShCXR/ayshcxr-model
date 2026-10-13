# Quarantined: FiLM symptom-fusion (label-leaky)

Moved here on 2026-10-13. **Do not report any number from these files as a model result.**

| File | What it is |
|---|---|
| `eval_film_simulation.py` | Evaluation of FiLM symptom fusion |
| `film_ablation_results.csv` | Its ablation results |
| `film_fusion_best_auc0.95–0.97*.pth` | FiLM checkpoints (weights are gitignored; local only) |

## Why these are quarantined

The symptom vectors used for training and validation were **generated from the disease
labels** (`SYMPTOM_PREVALENCE_BY_DISEASE` in `medical_knowledge.py`). The symptoms therefore
encoded the answer, which is label leakage. The 0.95–0.97 AUCs in the filenames are artifacts
of that leakage. With label-independent placeholder symptoms, test AUC was 0.6878
(see `docs/PROJECT_COMPLETE_SUMMARY.md` §1.2).

The live app does not load these checkpoints (`core/app.py`, FiLM retired: `film_model = None`).

A legitimate version needs real symptom inputs collected independently of the labels
(e.g. MIMIC-IV-ED triage data or PHC-collected checklists). It remains future work.
