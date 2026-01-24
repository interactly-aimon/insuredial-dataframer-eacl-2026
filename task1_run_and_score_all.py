#!/usr/bin/env python3
# task1_run_all_models.py — one-stop Task-1 evaluator for Gemini + OpenAI + Anthropic models
# ---------------------------------------------------------------------------
#  pip install "google-generativeai>=0.3" "openai>=1.25" "anthropic" "pydantic>=2.7" \
#               tenacity pandas tabulate tqdm python-dotenv
# ---------------------------------------------------------------------------

import os, glob, json, time, sys, math, asyncio, argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

# 3rd-party libs
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, ConfigDict, conlist
from dotenv import load_dotenv

# ----------------------------- DATA / PROMPT -----------------------------
DATA_DIR     = "data/real"
PROMPT_FILE  = "prompt_task1.txt"     # must contain {transcript_here}
PROMPT_TEMPLATE = Path(PROMPT_FILE).read_text()

# Synthetic data configuration
SYNTH_BASE_DIR = "data/synthetic"

# -------------------------------------------------------------------------
#  MODEL LISTS – edit here only
# -------------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

ANTHROPIC_MODELS = [
]

# ------------------------------ CONSTANTS --------------------------------
FIELDS = [
    "ivr", "greeting", "patient_info", "coverage_check",
    "drug1_formulary", "drug1_restrictions", "drug1_copay",
    "drug2_formulary", "drug2_restrictions", "drug2_copay",
    "agent_interaction",
]
_span_ok = lambda s: isinstance(s, list) and len(s) == 2 and s[0] <= s[1]

# ========================== SCORING HELPERS ==============================
def span_f1(g, p):
    if not (_span_ok(g) and _span_ok(p)):
        return 1.0 if (g is None and p is None) else 0.0
    gset, pset = set(range(g[0], g[1] + 1)), set(range(p[0], p[1] + 1))
    if not gset & pset:
        return 0.0
    prec = len(gset & pset) / len(pset)
    rec  = len(gset & pset) / len(gset)
    return 2 * prec * rec / (prec + rec)

def loose_exact(phase, g, p):
    """Loose EM: IVR=end+overlap, Agent=start±1, all others strict."""
    if phase == "ivr":
        if not (_span_ok(g) and _span_ok(p)):
            return g is None and p is None
        return (g[1] == p[1]) and (p[0] <= g[1])
    if phase == "agent_interaction":
        if not (_span_ok(g) and _span_ok(p)):
            return g is None and p is None
        return abs(g[0] - p[0]) <= 1
    return g == p

def span_sad(g, p):
    """Sum of absolute deviations for start and end positions. Skip if either is null."""
    if not (_span_ok(g) and _span_ok(p)):
        return None  # Skip null spans - no deviation calculated
    return abs(g[0] - p[0]) + abs(g[1] - p[1])

# ==================== DUAL NULL-HANDLING APPROACHES ======================
def loose_exact_skip_gold_null(phase, g, p):
    """Skip gold null version: returns None when gold=null (excluded from aggregation)"""
    if g is None:
        return None  # Skip from calculation
    # Rest same as current loose_exact()
    if phase == "ivr":
        if not (_span_ok(g) and _span_ok(p)):
            return p is None  # Only correct if pred also null
        return (g[1] == p[1]) and (p[0] <= g[1])
    if phase == "agent_interaction":
        if not (_span_ok(g) and _span_ok(p)):
            return p is None  # Only correct if pred also null
        return abs(g[0] - p[0]) <= 1
    return g == p

def loose_exact_include_gold_null(phase, g, p):
    """Include gold null version: null vs null = correct"""
    if g is None and p is None:
        return True  # Count as correct
    if g is None and p is not None:
        return False  # Count as incorrect
    # Rest same as current loose_exact()
    if phase == "ivr":
        if not (_span_ok(g) and _span_ok(p)):
            return False
        return (g[1] == p[1]) and (p[0] <= g[1])
    if phase == "agent_interaction":
        if not (_span_ok(g) and _span_ok(p)):
            return False
        return abs(g[0] - p[0]) <= 1
    return g == p

def span_f1_skip_gold_null(g, p):
    """Skip gold null version: returns None when gold=null"""
    if g is None:
        return None
    # Rest same as current span_f1()
    if not (_span_ok(g) and _span_ok(p)):
        return 0.0  # If gold valid but pred invalid
    gset, pset = set(range(g[0], g[1] + 1)), set(range(p[0], p[1] + 1))
    if not gset & pset:
        return 0.0
    prec = len(gset & pset) / len(pset)
    rec  = len(gset & pset) / len(gset)
    return 2 * prec * rec / (prec + rec)

def span_f1_include_gold_null(g, p):
    """Include gold null version: null vs null = 1.0"""
    if g is None and p is None:
        return 1.0
    if g is None and p is not None:
        return 0.0
    # Rest same as current span_f1()
    if not (_span_ok(g) and _span_ok(p)):
        return 0.0
    gset, pset = set(range(g[0], g[1] + 1)), set(range(p[0], p[1] + 1))
    if not gset & pset:
        return 0.0
    prec = len(gset & pset) / len(pset)
    rec  = len(gset & pset) / len(gset)
    return 2 * prec * rec / (prec + rec)

def safe_turn_range(drugs, idx, key):
    try:    return drugs[idx][key]["turn_range"]
    except (IndexError, KeyError, TypeError): return None

def safe_annotation_access(ann, phase_path, business_rule=None):
    """Extract turn_range with business rule validation and [0,0] artifact handling"""
    try:
        # Navigate to turn_range
        obj = ann
        for key in phase_path:
            obj = obj[key]
        
        # Apply business rule if provided
        if business_rule and not business_rule(ann):
            return None
            
        # Handle [0,0] artifacts - these represent skipped phases
        if obj == [0, 0]:
            return None
            
        return obj
    except (KeyError, IndexError, TypeError):
        return None

def coverage_check_rule(ann):
    """Coverage check only valid if patient record was found"""
    return ann.get("patient_info", {}).get("record_found", False)

def drug_restrictions_rule(ann, drug_idx):
    """Drug restrictions only valid if formulary = YES"""
    try:
        return ann["drugs"][drug_idx]["formulary"]["value"] == "YES"
    except (KeyError, IndexError):
        return False

def drug_copay_rule(ann, drug_idx):
    """Drug copay only valid if formulary = YES and restrictions explicitly = NO"""
    try:
        return (ann["drugs"][drug_idx]["formulary"]["value"] == "YES" and 
                ann["drugs"][drug_idx]["restrictions"]["value"] == "NO")
    except (KeyError, IndexError):
        return False

def discover_data_sources(data_source, sample_limit=None):
    """Discover available data files and generating models for evaluation."""
    if data_source == "real":
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
        if sample_limit:
            files = files[:sample_limit]
        return [("real", files)]
    
    elif data_source == "synthetic":
        data_sources = []
        files = sorted([f for f in glob.glob(os.path.join(SYNTH_BASE_DIR, "*.json")) 
                        if "sample_" in os.path.basename(f)])
        if sample_limit:
            files = files[:sample_limit]
        data_sources.append(('default', files))  # use 'default' as placeholder for generating model
        
        return data_sources
    
    else:
        print(f"❌ Unknown data source: {data_source}")
        return []

# ====================== GEMINI (Google Generative AI) =====================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "<key>"
from google import genai
from google.genai import types
if GOOGLE_API_KEY and GOOGLE_API_KEY != "<PASTE_GOOGLE_KEY_HERE>":
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    gemini_client = None

async def ask_gemini_async(model_name: str, transcript: dict):
    prompt = PROMPT_TEMPLATE.replace(
        "{transcript_here}", json.dumps(transcript, ensure_ascii=False, indent=2)
    )
    resp = await gemini_client.aio.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CallSpans,
            temperature=0.0
        )
    )
    try:
        out = json.loads(resp.text)
    except json.JSONDecodeError:
        out = {}
    meta = {
        "elapsed": getattr(resp, "_response", None)
                   and getattr(resp._response, "elapsed", None),
        "usage": str(getattr(resp, "usage_metadata", ""))
    }
    return out, meta

# =========================== OPENAI SET-UP ===============================
load_dotenv()   # allows .env file
import openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌  Set OPENAI_API_KEY env-var (or .env) to use OpenAI models.")
    OPENAI_MODELS.clear()       # avoid accidental calls
else:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# pydantic schema identical to your original script
Span = conlist(int, min_length=2, max_length=2)
class CallSpans(BaseModel):
    ivr:                Optional[Span] = None
    greeting:           Optional[Span] = None
    patient_info:       Optional[Span] = None
    coverage_check:     Optional[Span] = None
    drug1_formulary:    Optional[Span] = None
    drug1_restrictions: Optional[Span] = None
    drug1_copay:        Optional[Span] = None
    drug2_formulary:    Optional[Span] = None
    drug2_restrictions: Optional[Span] = None
    drug2_copay:        Optional[Span] = None
    agent_interaction:  Optional[Span] = None

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def ask_openai(model_name: str, transcript: dict):
    prompt = PROMPT_TEMPLATE.replace(
        "{transcript_here}", json.dumps(transcript, ensure_ascii=False, indent=2)
    )
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": CallSpans,
    }
    if not model_name.startswith("o3"):
        kwargs["temperature"] = 0
    resp = client.chat.completions.parse(**kwargs)
    return resp.choices[0].message.parsed.model_dump()

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def ask_openai_async(model_name: str, transcript: dict):
    prompt = PROMPT_TEMPLATE.replace(
        "{transcript_here}", json.dumps(transcript, ensure_ascii=False, indent=2)
    )
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": CallSpans,
    }
    if not model_name.startswith("o3"):
        kwargs["temperature"] = 0
    resp = await async_client.chat.completions.parse(**kwargs)
    return resp.choices[0].message.parsed.model_dump()

# ========================= ANTHROPIC SET-UP ==============================
import anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("❌  Set ANTHROPIC_API_KEY env-var (or .env) to use Anthropic models.")
    ANTHROPIC_MODELS.clear()
else:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    anthropic_async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def ask_anthropic_async(model_name: str, transcript: dict):
    prompt = PROMPT_TEMPLATE.replace(
        "{transcript_here}", json.dumps(transcript, ensure_ascii=False, indent=2)
    )
    tools = [{
        "name": "extract_call_spans",
        "description": "Extract turn range spans for each phase of the call",
        "input_schema": CallSpans.model_json_schema()
    }]
    resp = await anthropic_async_client.messages.create(
        model=model_name,
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_call_spans"},
        messages=[{"role": "user", "content": prompt}]
    )
    for content in resp.content:
        if content.type == "tool_use" and content.name == "extract_call_spans":
            return content.input
    return {}

async def process_files_for_model(model_name: str, provider: str, semaphore: asyncio.Semaphore, 
                                   data_source: str, generating_model: str = None, files: list = None):
    # Determine output paths based on data source
    if data_source == "real":
        output_dir = "real"
        pred_dir = f"real/predictions_task1_{model_name}"
        csv_out  = f"real/task1_accuracy_{model_name}.csv"
        sum_out  = f"real/task1_summary_{model_name}.json"
    else:
        # Synthetic data
        output_dir = "synth"
        pred_dir = f"synth/predictions_task1_{model_name}"
        csv_out  = f"synth/task1_accuracy_{model_name}.csv"
        sum_out  = f"synth/task1_summary_{model_name}.json"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    if generating_model:
        print(f"\n=== {provider.upper()} : {model_name} [GEN: {generating_model}] ===")
    else:
        print(f"\n=== {provider.upper()} : {model_name} ===")
    
    async def process_single_file(fpath):
        async with semaphore:  # Rate limiting per API
            stem  = Path(fpath).stem
            rec   = json.load(open(fpath))
            script_only = {"script": rec["script"]}

            # GOLD ----------------------------
            ann = rec["annotation"]
            gold = {
                "ivr": safe_annotation_access(ann, ["ivr", "turn_range"]),
                "greeting": safe_annotation_access(ann, ["greeting", "turn_range"]),
                "patient_info": safe_annotation_access(ann, ["patient_info", "turn_range"]),
                "coverage_check": safe_annotation_access(ann, ["coverage_check", "turn_range"], coverage_check_rule),
                "drug1_formulary": safe_annotation_access(ann, ["drugs", 0, "formulary", "turn_range"]),
                "drug1_restrictions": safe_annotation_access(ann, ["drugs", 0, "restrictions", "turn_range"], 
                                                            lambda a: drug_restrictions_rule(a, 0)),
                "drug1_copay": safe_annotation_access(ann, ["drugs", 0, "copay", "turn_range"], 
                                                     lambda a: drug_copay_rule(a, 0)),
                "drug2_formulary": safe_annotation_access(ann, ["drugs", 1, "formulary", "turn_range"]),
                "drug2_restrictions": safe_annotation_access(ann, ["drugs", 1, "restrictions", "turn_range"], 
                                                            lambda a: drug_restrictions_rule(a, 1)),
                "drug2_copay": safe_annotation_access(ann, ["drugs", 1, "copay", "turn_range"], 
                                                     lambda a: drug_copay_rule(a, 1)),
                "agent_interaction": safe_annotation_access(ann, ["agent_interaction", "turn_range"]),
            }

            # PREDICTION (cached) -------------
            cache = Path(pred_dir, f"{stem}_{model_name}.json")
            if cache.exists():
                pred = json.load(open(cache))["prediction"]
            else:
                if provider == "gemini":
                    pred, meta = await ask_gemini_async(model_name, script_only)
                elif provider == "anthropic":
                    try:
                        t0 = time.time()
                        pred = await ask_anthropic_async(model_name, script_only)
                        meta = {"elapsed": time.time() - t0}
                    except Exception as e:
                        print(f"{stem}: Anthropic call failed → {e}")
                        pred, meta = {}, {}
                else:   # openai
                    try:
                        t0 = time.time()
                        pred = await ask_openai_async(model_name, script_only)
                        meta = {"elapsed": time.time() - t0}
                    except Exception as e:
                        print(f"{stem}: OpenAI call failed → {e}")
                        pred, meta = {}, {}
                json.dump({"prediction": pred, "meta": meta}, open(cache, "w"), indent=2)

            # normalise malformed spans
            for k in FIELDS:
                if not _span_ok(pred.get(k)):
                    pred[k] = None

            # SCORING ------------------------
            # Skip gold null approach
            em_hits_skip = f1_hits_skip = 0.0
            em_count_skip = f1_count_skip = 0
            all_correct_skip = True
            
            # Include gold null approach  
            em_hits_include = f1_hits_include = 0.0
            all_correct_include = True
            
            # SAD (unchanged - always skips nulls)
            sad_total, sad_count = 0.0, 0
            
            per_field_results = {}
            for k in FIELDS:
                g, p = gold[k], pred.get(k)
                
                # Skip gold null approach
                em_skip = loose_exact_skip_gold_null(k, g, p)
                f1_skip = span_f1_skip_gold_null(g, p)
                
                if em_skip is not None:
                    em_count_skip += 1
                    if em_skip:
                        em_hits_skip += 1
                    else:
                        all_correct_skip = False
                
                if f1_skip is not None:
                    f1_count_skip += 1
                    f1_hits_skip += f1_skip
                
                # Include gold null approach
                em_include = loose_exact_include_gold_null(k, g, p)
                f1_include = span_f1_include_gold_null(g, p)
                
                if em_include:
                    em_hits_include += 1
                else:
                    all_correct_include = False
                    
                f1_hits_include += f1_include
                
                # SAD (unchanged)
                sad = span_sad(g, p)
                if sad is not None:
                    sad_total += sad
                    sad_count += 1
                
                per_field_results[k] = {
                    "em_skip": em_skip, "f1_skip": f1_skip,
                    "em_include": em_include, "f1_include": f1_include,
                    "sad": sad
                }

            return {
                "file": stem,
                # Skip gold null approach
                "Loose-EM-Skip-Gold-Null %": round(em_hits_skip / em_count_skip * 100, 2) if em_count_skip > 0 else 0.0,
                "F1-Skip-Gold-Null %": round(f1_hits_skip / f1_count_skip * 100, 2) if f1_count_skip > 0 else 0.0,
                "EM-Count-Skip": em_count_skip,
                "F1-Count-Skip": f1_count_skip,
                "all_correct_skip": all_correct_skip,
                # Include gold null approach
                "Loose-EM-Include-Gold-Null %": round(em_hits_include / len(FIELDS) * 100, 2),
                "F1-Include-Gold-Null %": round(f1_hits_include / len(FIELDS) * 100, 2),
                "all_correct_include": all_correct_include,
                # SAD (unchanged)
                "SAD Total": round(sad_total, 2),
                "SAD Count": sad_count,
                "per_field_results": per_field_results
            }

    # Process all files concurrently with semaphore control
    file_paths = files if files else sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    tasks = [process_single_file(fpath) for fpath in file_paths]
    
    # Use tqdm for progress tracking
    from tqdm.asyncio import tqdm as async_tqdm
    results = await async_tqdm.gather(*tasks, desc=model_name, leave=False)
    
    rows = []
    # Skip gold null aggregation
    em_skip_sum, em_skip_cnt = defaultdict(int), defaultdict(int)
    f1_skip_sum, f1_skip_cnt = defaultdict(float), defaultdict(int)
    
    # Include gold null aggregation
    em_include_sum, em_include_cnt = defaultdict(int), defaultdict(int)
    f1_include_sum = defaultdict(float)
    
    # SAD aggregation (unchanged)
    sad_sum, sad_cnt = defaultdict(float), defaultdict(int)
    
    for result in results:
        rows.append({
            "file": result["file"],
            "Loose-EM-Skip-Gold-Null %": result["Loose-EM-Skip-Gold-Null %"],
            "F1-Skip-Gold-Null %": result["F1-Skip-Gold-Null %"],
            "EM-Count-Skip": result["EM-Count-Skip"],
            "F1-Count-Skip": result["F1-Count-Skip"],
            "all_correct_skip": result["all_correct_skip"],
            "Loose-EM-Include-Gold-Null %": result["Loose-EM-Include-Gold-Null %"],
            "F1-Include-Gold-Null %": result["F1-Include-Gold-Null %"],
            "all_correct_include": result["all_correct_include"],
            "SAD Total": result["SAD Total"],
            "SAD Count": result["SAD Count"]
        })
        
        # Aggregate per-field stats
        for k, field_result in result["per_field_results"].items():
            # Skip gold null approach
            if field_result["em_skip"] is not None:
                em_skip_cnt[k] += 1
                if field_result["em_skip"]:
                    em_skip_sum[k] += 1
            
            if field_result["f1_skip"] is not None:
                f1_skip_cnt[k] += 1
                f1_skip_sum[k] += field_result["f1_skip"]
            
            # Include gold null approach
            em_include_cnt[k] += 1
            if field_result["em_include"]:
                em_include_sum[k] += 1
            f1_include_sum[k] += field_result["f1_include"]
            
            # SAD (unchanged)
            if field_result["sad"] is not None:
                sad_sum[k] += field_result["sad"]
                sad_cnt[k] += 1

    # --------- corpus summaries ----------
    # Skip gold null tables
    phase_em_skip_tbl = [[k, f"{em_skip_sum[k] / em_skip_cnt[k] * 100:6.1f}%" if em_skip_cnt[k] > 0 else "N/A"] for k in FIELDS]
    phase_f1_skip_tbl = [[k, f"{f1_skip_sum[k] / f1_skip_cnt[k] * 100:6.1f}%" if f1_skip_cnt[k] > 0 else "N/A"] for k in FIELDS]
    
    # Include gold null tables
    phase_em_include_tbl = [[k, f"{em_include_sum[k] / em_include_cnt[k] * 100:6.1f}%"] for k in FIELDS]
    phase_f1_include_tbl = [[k, f"{f1_include_sum[k] / em_include_cnt[k] * 100:6.1f}%"] for k in FIELDS]
    
    # SAD table (unchanged)
    phase_sad_tbl = [[k, f"{sad_sum[k] / sad_cnt[k]:6.1f}" if sad_cnt[k] > 0 else "N/A"] for k in FIELDS]

    # Corpus-level metrics
    # Skip gold null approach
    corpus_em_skip = sum(r["Loose-EM-Skip-Gold-Null %"] for r in rows) / len(rows)
    corpus_f1_skip = sum(r["F1-Skip-Gold-Null %"] for r in rows) / len(rows)
    call_em_skip_count = sum(r["all_correct_skip"] for r in rows)
    call_em_skip_pct = call_em_skip_count / len(rows) * 100
    
    # Include gold null approach  
    corpus_em_include = sum(r["Loose-EM-Include-Gold-Null %"] for r in rows) / len(rows)
    corpus_f1_include = sum(r["F1-Include-Gold-Null %"] for r in rows) / len(rows)
    call_em_include_count = sum(r["all_correct_include"] for r in rows)
    call_em_include_pct = call_em_include_count / len(rows) * 100
    
    # SAD (unchanged)
    corpus_sad = sum(r["SAD Total"] for r in rows) / len(rows)
    n_calls = len(rows)

    # pretty print - Skip Gold Null approach
    print("\n=== SKIP GOLD NULL APPROACH (strict evaluation) ===")
    print(tabulate(phase_em_skip_tbl, headers=["Phase", "Loose EM (Skip)"], tablefmt="github"))
    print(tabulate(phase_f1_skip_tbl, headers=["Phase", "F1 (Skip)"], tablefmt="github"))
    print(f"Corpus EM (slot-level, skip): {corpus_em_skip:5.1f}%")
    print(f"Corpus F1 (slot-level, skip): {corpus_f1_skip:5.1f}%")
    print(f"Corpus SAD (sum abs deviations): {corpus_sad:5.1f}")
    print(f"Call-level Loose EM (skip gold null): {call_em_skip_count}/{n_calls} ({call_em_skip_pct:.1f}%)")
    
    # pretty print - Include Gold Null approach
    print("\n=== INCLUDE GOLD NULL APPROACH (inclusive evaluation) ===")
    print(tabulate(phase_em_include_tbl, headers=["Phase", "Loose EM (Include)"], tablefmt="github"))
    print(tabulate(phase_f1_include_tbl, headers=["Phase", "F1 (Include)"], tablefmt="github"))
    print(f"Corpus EM (slot-level, include): {corpus_em_include:5.1f}%")
    print(f"Corpus F1 (slot-level, include): {corpus_f1_include:5.1f}%")
    print(f"Call-level Loose EM (include gold null): {call_em_include_count}/{n_calls} ({call_em_include_pct:.1f}%)")
    
    # SAD table (unchanged)
    print("\n=== SAD METRICS (unchanged) ===")
    print(tabulate(phase_sad_tbl, headers=["Phase", "SAD Mean"], tablefmt="github"))

    # save artefacts
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    json.dump({
        # Skip gold null approach
        "phase_accuracy_loose_skip_gold_null": {k: v for k, v in zip(FIELDS, [v[1] for v in phase_em_skip_tbl])},
        "phase_f1_scores_skip_gold_null": {k: v for k, v in zip(FIELDS, [v[1] for v in phase_f1_skip_tbl])},
        "corpus_slot_EM_loose_skip_gold_null": corpus_em_skip,
        "corpus_slot_F1_skip_gold_null": corpus_f1_skip,
        "call_level_loose_em_count_skip_gold_null": int(call_em_skip_count),
        "call_level_loose_em_pct_skip_gold_null": call_em_skip_pct,
        
        # Include gold null approach
        "phase_accuracy_loose_include_gold_null": {k: v for k, v in zip(FIELDS, [v[1] for v in phase_em_include_tbl])},
        "phase_f1_scores_include_gold_null": {k: v for k, v in zip(FIELDS, [v[1] for v in phase_f1_include_tbl])},
        "corpus_slot_EM_loose_include_gold_null": corpus_em_include,
        "corpus_slot_F1_include_gold_null": corpus_f1_include,
        "call_level_loose_em_count_include_gold_null": int(call_em_include_count),
        "call_level_loose_em_pct_include_gold_null": call_em_include_pct,
        
        # SAD (unchanged)
        "phase_sad_means": {k: v for k, v in zip(FIELDS, [v[1] for v in phase_sad_tbl])},
        "corpus_sad_mean": corpus_sad
    }, open(sum_out, "w"), indent=2)

    print(f"↑ CSV   → {csv_out}\n↑ JSON  → {sum_out}\n↑ cache → {pred_dir}/")

async def run_all_models(data_source: str, sample_limit: int = None, parallel: bool = False):
    # Discover data sources
    data_sources = discover_data_sources(data_source, sample_limit)
    if not data_sources:
        print(f"❌ No data sources found for: {data_source}")
        return

    # Separate semaphores for each API's rate limits
    gemini_semaphore = asyncio.Semaphore(50)
    openai_semaphore = asyncio.Semaphore(50)
    anthropic_semaphore = asyncio.Semaphore(50)

    # Process each data source (generating model or real data)
    for generating_model, files in data_sources:
        print(f"\n🔍 Processing data source: {generating_model} ({len(files)} files)")

        async def run_gemini_models():
            for model in GEMINI_MODELS:
                if not GOOGLE_API_KEY or GOOGLE_API_KEY == "<PASTE_GOOGLE_KEY_HERE>":
                    print("\n⚠︎ Skipping Gemini models — set GOOGLE_API_KEY or edit the script.\n")
                    break
                await process_files_for_model(model, "gemini", gemini_semaphore,
                                             data_source, generating_model, files)

        async def run_openai_models():
            for model in OPENAI_MODELS:
                await process_files_for_model(model, "openai", openai_semaphore,
                                             data_source, generating_model, files)

        async def run_anthropic_models():
            for model in ANTHROPIC_MODELS:
                await process_files_for_model(model, "anthropic", anthropic_semaphore,
                                             data_source, generating_model, files)

        if parallel:
            # Run all APIs in parallel with their own rate limits
            await asyncio.gather(
                run_gemini_models(),
                run_openai_models(),
                run_anthropic_models()
            )
        else:
            # Run APIs sequentially to avoid output interleaving
            await run_gemini_models()
            await run_openai_models()
            await run_anthropic_models()

# ============================== MAIN ====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 Evaluation: Phase Boundary Detection")
    parser.add_argument("--data-source", choices=["real", "synthetic"],
                       default="real", help="Data source to evaluate")
    parser.add_argument("--sample-limit", type=int, default=5,
                       help="Limit number of samples per generating model (default: 5)")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True,
                       help="Run API calls in parallel (default: True)")

    args = parser.parse_args()

    print(f"🚀 Starting Task 1 evaluation with data source: {args.data_source}")
    print(f"📊 Sample limit: {args.sample_limit}")
    print(f"⚙️  Execution mode: {'parallel' if args.parallel else 'sequential'}")

    # Run async version
    asyncio.run(run_all_models(args.data_source, args.sample_limit, args.parallel))
