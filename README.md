# INSURE-Dial

**INSURE-Dial: A Phase-Aware Conversational Dataset & Benchmark for Compliance Verification and Phase Detection** (EACL 2026)

Shubham Kulkarni, Shiva Chaitanya (Interactly.ai)

Alexander Lyzhov, Preetam Joshi (AIMon Labs)

## Evaluation Scripts

Two scripts for evaluating models on insurance call transcripts:
- `task1_run_and_score_all.py` — span extraction (turn ranges), scored with loose EM + F1
- `task2_run_all_models.py` — compliance evaluation (IC/PC), scored with accuracy

Both run Gemini, OpenAI, and Anthropic models in parallel. Missing API keys skip that provider.

## Setup

```bash
pip install google-generativeai openai anthropic pydantic tenacity pandas tabulate tqdm python-dotenv

export GOOGLE_API_KEY="..."      # optional
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Usage

```bash
python task1_run_and_score_all.py --data-source real --sample-limit 5
python task2_run_all_models.py --data-source synthetic
```

| Option | Default | |
|--------|---------|--|
| `--data-source` | `real` | `real` or `synthetic` |
| `--sample-limit` | `5` | Number of files to process (0 = all) |

## Data

Input: `data/real/*.json` or `data/synthetic/*.json`

Each file contains `{"script": [...], "annotation": {...}}`

Prompts: `prompt_task1.txt`, `prompt_task2.txt` (must include `{transcript_here}` placeholder)

## Outputs

Results written to `real/` or `synth/` depending on data source:
```
real/
├── predictions_task1_<model>/   # cached per-call predictions
├── task1_accuracy_<model>.csv   # per-file metrics
└── task1_summary_<model>.json   # corpus-level summary
```

Predictions are cached — rerunning regenerates metrics without new API calls.

## Task-2 phases

Task-2 evaluates Information Compliance (IC) and Procedural Compliance (PC) across these phases:

| Phase | Description |
|-------|-------------|
| PID | Patient identification |
| CSV | Coverage status verification |
| DFV | Drug formulary (per drug) |
| DRC | Drug restrictions (per drug) |
| DCC | Drug copay (per drug) |
| CRN | Representative name (PC only) |
