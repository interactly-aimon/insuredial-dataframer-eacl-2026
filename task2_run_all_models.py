#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Evaluate Task-2 (IC & PC) on every Gemini + OpenAI + Anthropic model you list below.
#
#  pip install "google-generativeai>=0.3" "openai>=1.25" "anthropic" "pydantic>=2.7" \
#               tenacity pandas tabulate tqdm python-dotenv
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
import os, glob, json, time, math, sys, asyncio, argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, Literal, Union

import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, ConfigDict, conlist

# ---------------------------------------------------------------------------
#  DATA & PROMPT
# ---------------------------------------------------------------------------
DATA_DIR    = "data/real"
PROMPT_FILE = "prompt_task2.txt"      # must contain {transcript_here}
PROMPT_TEMPLATE = Path(PROMPT_FILE).read_text()

# Synthetic data configuration
SYNTH_BASE_DIR = "data/synthetic"

# ---------------------------------------------------------------------------
#  MODEL LISTS – edit here only
# ---------------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",

    "gemini-2.5-pro",
]

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",

    "gpt-4.1",
    "o3",
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-0",
]

# ===================== GOLD-LABEL LOGIC (unchanged) =========================
_span_ok = lambda s: isinstance(s, list) and len(s) == 2 and s[0] <= s[1]
_ic      = lambda ans, span:               ans and _span_ok(span)
_pc      = lambda ask, ans, span: ask and ans and _span_ok(span)

def _gold(ann: dict, strict: bool):
    """Return phase→gold dict for IC (strict=False) or PC (strict=True)."""
    try:
        pi, cc, drugs, ai = (ann["patient_info"], ann["coverage_check"],
                             ann["drugs"], ann["agent_interaction"])
    except (KeyError, TypeError):
        # Return all NA if annotation structure is corrupt
        phases = {
            "PID": "NA", "CSV": "NA",
            "DFV_drug_1": "NA", "DRC_drug_1": "NA", "DCC_drug_1": "NA",
            "DFV_drug_2": "NA", "DRC_drug_2": "NA", "DCC_drug_2": "NA",
        }
        if strict:
            phases["CRN"] = "NA"
        phases["overall"] = True  # All NA counts as correct
        return phases
    
    chk = _pc if strict else _ic

    try:
        pid = chk(pi["user_asked_details"],
                  pi["assistant_provided_details"], pi["turn_range"]) if strict else \
              _ic(pi["assistant_provided_details"], pi["turn_range"])
    except (KeyError, TypeError):
        pid = "NA"

    try:
        if not pi["record_found"]:
            csv = "NA"
        else:
            csv = chk(cc["assistant_asked_status"],
                      cc["user_answered_status"], cc["turn_range"]) if strict else \
                  _ic(cc["user_answered_status"], cc["turn_range"])
    except (KeyError, TypeError):
        csv = "NA"

    # Initialize with defaults for 2 drugs
    dfv, drc, dcc = ["NA", "NA"], ["NA", "NA"], ["NA", "NA"]
    
    # Process up to 2 drugs, handling corruption gracefully
    for i in range(2):
        try:
            d = drugs[i] if i < len(drugs) else None
            if d is None:
                continue  # Keep defaults for missing drugs
                
            if csv is not True or cc.get("plan_status") != "ACTIVE":
                dfv[i] = "NA"
            else:
                dfv[i] = (
                    chk(d["formulary"]["assistant_asked"],
                        d["formulary"]["user_answered"],
                        d["formulary"]["turn_range"]) if strict else
                    _ic(d["formulary"]["user_answered"],
                        d["formulary"]["turn_range"])
                )
            
            if dfv[i] is True and d.get("formulary", {}).get("value") == "YES":
                drc[i] = (
                    chk(d["restrictions"]["assistant_asked"],
                        d["restrictions"]["user_answered"],
                        d["restrictions"]["turn_range"]) if strict else
                    _ic(d["restrictions"]["user_answered"],
                        d["restrictions"]["turn_range"])
                )
            else:
                drc[i] = "NA"
                
            need_dcc = (drc[i] is True) and d.get("restrictions", {}).get("value") == "NO"
            if need_dcc:
                dcc[i] = (
                    chk(d["copay"]["assistant_asked"],
                        d["copay"]["user_answered"],
                        d["copay"]["turn_range"]) if strict else
                    _ic(d["copay"]["user_answered"],
                        d["copay"]["turn_range"])
                )
            else:
                dcc[i] = "NA"
                
        except (KeyError, IndexError, TypeError):
            # Keep defaults (NA) for corrupt drug data
            continue

    try:
        crn = (_pc(ai["assistant_asked_user_name"],
                   bool(ai["user_name_provided"]),
                   ai["turn_range"]) if strict else "NA")
    except (KeyError, TypeError):
        crn = "NA"

    phases = {
        "PID": pid, "CSV": csv,
        "DFV_drug_1": dfv[0], "DRC_drug_1": drc[0], "DCC_drug_1": dcc[0],
        "DFV_drug_2": dfv[1], "DRC_drug_2": drc[1], "DCC_drug_2": dcc[1],
    }
    if strict:
        phases["CRN"] = crn
    phases["overall"] = all(v is True or v == "NA" for v in phases.values())
    return phases

FIELDS_IC = ["PID","CSV","DFV_drug_1","DRC_drug_1","DCC_drug_1",
             "DFV_drug_2","DRC_drug_2","DCC_drug_2","overall"]
FIELDS_PC = FIELDS_IC[:-1] + ["CRN","overall"]

def discover_data_sources(data_source, sample_limit=None):
    """Discover available data files and generating models for evaluation."""
    if data_source == "real":
        files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
        if sample_limit:
            files = files[:sample_limit]
        return [("real", files)]

    elif data_source == "synthetic":
        files = sorted([f for f in glob.glob(os.path.join(SYNTH_BASE_DIR, "*.json"))
                        if "sample_" in os.path.basename(f)])
        if sample_limit:
            files = files[:sample_limit]
        return [('default', files)]

    else:
        print(f"❌ Unknown data source: {data_source}")
        return []

# ---------------------------------------------------------------------------
#  GEMINI CALL
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "key"
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
            response_schema=ComplianceSchema,
            temperature=0.0
        )
    )
    try:   out = json.loads(resp.text)
    except json.JSONDecodeError: out = {}
    meta = {"elapsed": getattr(resp,"_response",None)
                        and getattr(resp._response,"elapsed",None)}
    return out, meta

# ---------------------------------------------------------------------------
#  OPENAI CALL (Structured Output)
# ---------------------------------------------------------------------------
load_dotenv()
import openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌  Set OPENAI_API_KEY to use OpenAI models.")
    OPENAI_MODELS.clear()
else:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

BoolNA = Union[bool, Literal["NA"]]
class _Base(BaseModel): pass
class InfoCompliance(_Base):
    PID: BoolNA; CSV: BoolNA; DFV_drug_1: BoolNA; DRC_drug_1: BoolNA; DCC_drug_1: BoolNA
    DFV_drug_2: BoolNA; DRC_drug_2: BoolNA; DCC_drug_2: BoolNA; overall_IC: bool
class ProceduralCompliance(_Base):
    PID: BoolNA; CSV: BoolNA; DFV_drug_1: BoolNA; DRC_drug_1: BoolNA; DCC_drug_1: BoolNA
    DFV_drug_2: BoolNA; DRC_drug_2: BoolNA; DCC_drug_2: BoolNA; CRN: BoolNA; overall_PC: bool
class ComplianceSchema(_Base):
    Information_Compliance: InfoCompliance
    Procedural_Compliance:  ProceduralCompliance

@retry(wait=wait_random_exponential(1,20), stop=stop_after_attempt(6))
def ask_openai(model_name:str, transcript:dict):
    prompt = PROMPT_TEMPLATE.replace(
        "{transcript_here}", json.dumps(transcript, ensure_ascii=False, indent=2)
    )
    kwargs = {
        "model": model_name,
        "messages": [{"role":"user","content":prompt}],
        "response_format": ComplianceSchema,
    }
    if not model_name.startswith("o3"):
        kwargs["temperature"] = 0
    resp = client.chat.completions.parse(**kwargs)
    return resp.choices[0].message.parsed.model_dump()

@retry(wait=wait_random_exponential(1,20), stop=stop_after_attempt(6))
async def ask_openai_async(model_name:str, transcript:dict):
    prompt = PROMPT_TEMPLATE.replace(
        "{transcript_here}", json.dumps(transcript, ensure_ascii=False, indent=2)
    )
    kwargs = {
        "model": model_name,
        "messages": [{"role":"user","content":prompt}],
        "response_format": ComplianceSchema,
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

@retry(wait=wait_random_exponential(1,20), stop=stop_after_attempt(6))
async def ask_anthropic_async(model_name:str, transcript:dict):
    prompt = PROMPT_TEMPLATE.replace(
        "{transcript_here}", json.dumps(transcript, ensure_ascii=False, indent=2)
    )
    tools = [{
        "name": "extract_compliance",
        "description": "Extract information and procedural compliance metrics",
        "input_schema": ComplianceSchema.model_json_schema()
    }]
    resp = await anthropic_async_client.messages.create(
        model=model_name,
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_compliance"},
        messages=[{"role": "user", "content": prompt}]
    )
    for content in resp.content:
        if content.type == "tool_use" and content.name == "extract_compliance":
            return content.input
    return {}

# ---------------------------------------------------------------------------
#  UTILITIES
# ---------------------------------------------------------------------------
def _remap_ic(d):
    if "overall_IC" in d:
        d = d.copy(); d["overall"] = d.pop("overall_IC")
    return d
def _remap_pc(d):
    if "overall_PC" in d:
        d = d.copy(); d["overall"] = d.pop("overall_PC")
    return d
def _acc(ok, tot): return f"{ok/tot*100:5.1f}%" if tot else "NA"
_mean = lambda xs: sum(xs)/len(xs)
_stdev = lambda xs: math.sqrt(sum((x-_mean(xs))**2 for x in xs)/len(xs))

# ==================== DUAL NULL-HANDLING APPROACHES ======================
def compliance_skip_gold_na(gold_val, pred_val):
    """Skip gold NA version: returns None when gold=NA (excluded from aggregation)"""
    if gold_val == "NA":
        return None  # Skip from calculation
    return (gold_val == "NA" and pred_val in (None, "NA")) or pred_val == gold_val

def compliance_include_gold_na(gold_val, pred_val):
    """Include gold NA version: NA vs NA = correct (current behavior)"""
    return (gold_val == "NA" and pred_val in (None, "NA")) or pred_val == gold_val

def calculate_f1_score(true_positives, false_positives, false_negatives):
    """Calculate F1 score from confusion matrix components."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def calculate_per_phase_f1_scores(phase_data):
    """Calculate F1 scores for each phase, excluding NA cases."""
    f1_scores = {}
    for phase, data in phase_data.items():
        if phase == 'overall':
            continue  # Skip overall for F1 calculation
            
        # Count TP, FP, FN (excluding NA cases)
        tp = fp = fn = 0
        for pred, gt in data:
            if gt == "NA" or pred in (None, "NA"):
                continue  # Skip NA cases
            if pred == True and gt == True:
                tp += 1
            elif pred == True and gt == False:
                fp += 1
            elif pred == False and gt == True:
                fn += 1
        
        f1_scores[phase] = calculate_f1_score(tp, fp, fn) if (tp + fp + fn) > 0 else None
    
    return f1_scores

def calculate_per_phase_f1_scores_skip_gold_na(phase_data):
    """Calculate F1 scores for each phase, excluding gold=NA cases from calculation."""
    f1_scores = {}
    for phase, data in phase_data.items():
        if phase == 'overall':
            continue  # Skip overall for F1 calculation
            
        # Count TP, FP, FN (excluding gold=NA cases)
        tp = fp = fn = 0
        for pred, gt in data:
            if gt == "NA":
                continue  # Skip gold=NA cases entirely
            if pred in (None, "NA") and gt == "NA":
                continue  # Already skipped above
            if pred == True and gt == True:
                tp += 1
            elif pred == True and gt == False:
                fp += 1
            elif pred == False and gt == True:
                fn += 1
        
        f1_scores[phase] = calculate_f1_score(tp, fp, fn) if (tp + fp + fn) > 0 else None
    
    return f1_scores


# ---------------------------------------------------------------------------
#  ASYNC PARALLEL MAIN PER-MODEL RUNNER
# ---------------------------------------------------------------------------
async def process_files_for_model(model_name: str, provider: str, semaphore: asyncio.Semaphore, 
                                   data_source: str, generating_model: str = None, files: list = None):
    # Determine output paths based on data source
    if data_source == "real":
        output_dir = "real"
        pred_dir = f"real/predictions_task2_{model_name}"
        csv_out  = f"real/task2_accuracy_{model_name}.csv"
        sum_out  = f"real/task2_summary_{model_name}.json"
    else:
        # Synthetic data
        output_dir = "synth"
        pred_dir = f"synth/predictions_task2_{model_name}"
        csv_out  = f"synth/task2_accuracy_{model_name}.csv"
        sum_out  = f"synth/task2_summary_{model_name}.json"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(pred_dir).mkdir(exist_ok=True)

    if generating_model:
        print(f"\n=== {provider.upper()} : {model_name} [GEN: {generating_model}] ===")
    else:
        print(f"\n=== {provider.upper()} : {model_name} ===")
    
    async def process_single_file(call_path):
        async with semaphore:  # Rate limiting per API
            stem   = Path(call_path).stem
            record = json.load(open(call_path))
            transcript_only = {"script": record["script"]}

            cache_file = Path(pred_dir, f"{stem}_{model_name}.json")
            if cache_file.exists():
                pred = json.load(open(cache_file))["prediction"]
            else:
                if provider == "gemini":
                    pred, meta = await ask_gemini_async(model_name, transcript_only)
                elif provider == "anthropic":
                    try:
                        t0 = time.time()
                        pred = await ask_anthropic_async(model_name, transcript_only)
                        meta = {"elapsed": time.time()-t0}
                    except Exception as e:
                        print(f"{stem}: Anthropic API error → {e}")
                        pred, meta = {"Information_Compliance": {},
                                      "Procedural_Compliance": {}}, {}
                else:
                    try:
                        t0 = time.time()
                        pred = await ask_openai_async(model_name, transcript_only)
                        meta = {"elapsed": time.time()-t0}
                    except Exception as e:
                        print(f"{stem}: OpenAI API error → {e}")
                        pred, meta = {"Information_Compliance": {},
                                      "Procedural_Compliance": {}}, {}
                json.dump({"prediction": pred, "meta": meta}, open(cache_file,"w"), indent=2)

            gold_ic = _gold(record["annotation"], strict=False)
            gold_pc = _gold(record["annotation"], strict=True)

            pred_ic = _remap_ic(pred.get("Information_Compliance", {}))
            pred_pc = _remap_pc(pred.get("Procedural_Compliance",  {}))

            # Skip gold NA approach counters
            per_ic_skip = per_pc_skip = 0
            required_ic_skip = required_pc_skip = 0
            
            # Include gold NA approach counters  
            per_ic_include = per_pc_include = 0
            required_ic_include = len(FIELDS_IC)
            required_pc_include = len(FIELDS_PC)
            
            line = {"file": stem}
            
            # Store per-field results for aggregation - both approaches
            ic_results_skip = {}
            pc_results_skip = {}
            ic_results_include = {}
            pc_results_include = {}
            ic_phase_pairs_skip = {}
            pc_phase_pairs_skip = {}
            ic_phase_pairs_include = {}
            pc_phase_pairs_include = {}

            # IC scoring - dual approaches
            for k in FIELDS_IC:
                g, p = gold_ic[k], pred_ic.get(k)
                
                # Skip gold NA approach
                result_skip = compliance_skip_gold_na(g, p)
                if result_skip is not None:
                    required_ic_skip += 1
                    if result_skip:
                        per_ic_skip += 1
                    ic_results_skip[k] = result_skip
                    # Only store phase pairs for skip approach when not skipped
                    ic_phase_pairs_skip[k] = (p, g)
                else:
                    ic_results_skip[k] = None
                
                # Include gold NA approach
                result_include = compliance_include_gold_na(g, p)
                if result_include:
                    per_ic_include += 1
                ic_results_include[k] = result_include
                # Always store phase pairs for include approach
                ic_phase_pairs_include[k] = (p, g)
                
                line[f"IC_{k}"] = result_include  # Store include result for CSV

            # PC scoring - dual approaches
            for k in FIELDS_PC:
                g, p = gold_pc[k], pred_pc.get(k)
                
                # Skip gold NA approach
                result_skip = compliance_skip_gold_na(g, p)
                if result_skip is not None:
                    required_pc_skip += 1
                    if result_skip:
                        per_pc_skip += 1
                    pc_results_skip[k] = result_skip
                    # Only store phase pairs for skip approach when not skipped
                    pc_phase_pairs_skip[k] = (p, g)
                else:
                    pc_results_skip[k] = None
                
                # Include gold NA approach
                result_include = compliance_include_gold_na(g, p)
                if result_include:
                    per_pc_include += 1
                pc_results_include[k] = result_include
                # Always store phase pairs for include approach
                pc_phase_pairs_include[k] = (p, g)
                
                line[f"PC_{k}"] = result_include  # Store include result for CSV

            return {
                "line": line,
                # Skip gold NA approach
                "hit_rate_ic_skip": per_ic_skip/required_ic_skip if required_ic_skip > 0 else 0.0,
                "hit_rate_pc_skip": per_pc_skip/required_pc_skip if required_pc_skip > 0 else 0.0,
                "ic_results_skip": ic_results_skip,
                "pc_results_skip": pc_results_skip,
                # Include gold NA approach
                "hit_rate_ic_include": per_ic_include/required_ic_include,
                "hit_rate_pc_include": per_pc_include/required_pc_include,
                "ic_results_include": ic_results_include,
                "pc_results_include": pc_results_include,
                # Phase pairs separated by approach
                "ic_phase_pairs_skip": ic_phase_pairs_skip,
                "pc_phase_pairs_skip": pc_phase_pairs_skip,
                "ic_phase_pairs_include": ic_phase_pairs_include,
                "pc_phase_pairs_include": pc_phase_pairs_include
            }

    # Process all files concurrently with semaphore control
    file_paths = files if files else sorted(glob.glob(f"{DATA_DIR}/*.json"))
    tasks = [process_single_file(call_path) for call_path in file_paths]
    
    # Use tqdm for progress tracking
    from tqdm.asyncio import tqdm as async_tqdm
    results = await async_tqdm.gather(*tasks, desc=model_name, leave=False)
    
    # Aggregate results for both approaches
    rows = []
    # Skip gold NA approach
    tot_ic_skip, cnt_ic_skip, tot_pc_skip, cnt_pc_skip = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    hit_rates_ic_skip, hit_rates_pc_skip = [], []
    
    # Include gold NA approach
    tot_ic_include, cnt_ic_include, tot_pc_include, cnt_pc_include = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    hit_rates_ic_include, hit_rates_pc_include = [], []
    
    # Separate phase data for skip vs include approaches
    ic_phase_data_skip, pc_phase_data_skip = defaultdict(list), defaultdict(list)
    ic_phase_data_include, pc_phase_data_include = defaultdict(list), defaultdict(list)
    
    for result in results:
        rows.append(result["line"])
        
        # Skip gold NA approach hit rates
        hit_rates_ic_skip.append(result["hit_rate_ic_skip"])
        hit_rates_pc_skip.append(result["hit_rate_pc_skip"])
        
        # Include gold NA approach hit rates
        hit_rates_ic_include.append(result["hit_rate_ic_include"])
        hit_rates_pc_include.append(result["hit_rate_pc_include"])
        
        # Collect phase pairs for F1 calculation - separated by approach
        for k, pair in result["ic_phase_pairs_skip"].items():
            ic_phase_data_skip[k].append(pair)
        for k, pair in result["pc_phase_pairs_skip"].items():
            pc_phase_data_skip[k].append(pair)
        for k, pair in result["ic_phase_pairs_include"].items():
            ic_phase_data_include[k].append(pair)
        for k, pair in result["pc_phase_pairs_include"].items():
            pc_phase_data_include[k].append(pair)
        
        # Aggregate per-field stats - skip gold NA approach
        for k, is_correct in result["ic_results_skip"].items():
            if is_correct is not None:
                cnt_ic_skip[k] += 1
                if is_correct:
                    tot_ic_skip[k] += 1
                    
        for k, is_correct in result["pc_results_skip"].items():
            if is_correct is not None:
                cnt_pc_skip[k] += 1
                if is_correct:
                    tot_pc_skip[k] += 1
        
        # Aggregate per-field stats - include gold NA approach
        for k, is_correct in result["ic_results_include"].items():
            cnt_ic_include[k] += 1
            if is_correct:
                tot_ic_include[k] += 1
                
        for k, is_correct in result["pc_results_include"].items():
            cnt_pc_include[k] += 1
            if is_correct:
                tot_pc_include[k] += 1

    # ----- Skip Gold NA Approach Tables -----
    print("\n=== SKIP GOLD NA APPROACH (strict evaluation) ===")
    table_skip = []
    for k in FIELDS_IC: table_skip.append(["IC", k, _acc(tot_ic_skip[k], cnt_ic_skip[k])])
    for k in FIELDS_PC: table_skip.append(["PC", k, _acc(tot_pc_skip[k], cnt_pc_skip[k])])
    print(tabulate(table_skip, headers=["Block","Phase","Accuracy (Skip)"], tablefmt="github"))
    
    # ----- Include Gold NA Approach Tables -----  
    print("\n=== INCLUDE GOLD NA APPROACH (inclusive evaluation) ===")
    table_include = []
    for k in FIELDS_IC: table_include.append(["IC", k, _acc(tot_ic_include[k], cnt_ic_include[k])])
    for k in FIELDS_PC: table_include.append(["PC", k, _acc(tot_pc_include[k], cnt_pc_include[k])])
    print(tabulate(table_include, headers=["Block","Phase","Accuracy (Include)"], tablefmt="github"))

    # Skip gold NA approach hit rates
    m_ic_skip = _mean(hit_rates_ic_skip)*100 if hit_rates_ic_skip else 0.0
    s_ic_skip = _stdev(hit_rates_ic_skip)*100 if len(hit_rates_ic_skip) > 1 else 0.0
    m_pc_skip = _mean(hit_rates_pc_skip)*100 if hit_rates_pc_skip else 0.0
    s_pc_skip = _stdev(hit_rates_pc_skip)*100 if len(hit_rates_pc_skip) > 1 else 0.0
    
    # Include gold NA approach hit rates
    m_ic_include = _mean(hit_rates_ic_include)*100 if hit_rates_ic_include else 0.0
    s_ic_include = _stdev(hit_rates_ic_include)*100 if len(hit_rates_ic_include) > 1 else 0.0
    m_pc_include = _mean(hit_rates_pc_include)*100 if hit_rates_pc_include else 0.0
    s_pc_include = _stdev(hit_rates_pc_include)*100 if len(hit_rates_pc_include) > 1 else 0.0
    
    print(f"\nPer-call hit-rate (Skip)   IC {m_ic_skip:5.1f}% ± {s_ic_skip:.1f}")
    print(f"Per-call hit-rate (Skip)   PC {m_pc_skip:5.1f}% ± {s_pc_skip:.1f}")
    print(f"Per-call hit-rate (Include) IC {m_ic_include:5.1f}% ± {s_ic_include:.1f}")
    print(f"Per-call hit-rate (Include) PC {m_pc_include:5.1f}% ± {s_pc_include:.1f}")
    
    # Calculate F1 scores for skip approach only
    ic_f1_scores_skip = calculate_per_phase_f1_scores_skip_gold_na(ic_phase_data_skip)
    pc_f1_scores_skip = calculate_per_phase_f1_scores_skip_gold_na(pc_phase_data_skip)
    
    # Calculate macro-averaged metrics
    def macro_average_valid(scores):
        valid_scores = [score for score in scores.values() if score is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else None
    
    # Skip gold NA macro averages
    macro_f1_ic_skip = macro_average_valid(ic_f1_scores_skip)
    macro_f1_pc_skip = macro_average_valid(pc_f1_scores_skip)
    
    # Extract per-phase accuracies for macro-averaging (skip approach)
    ic_accuracies_skip = {k: float(v.rstrip('%')) for b, k, v in table_skip if b == "IC" and k != "overall" and v != "NA"}
    pc_accuracies_skip = {k: float(v.rstrip('%')) for b, k, v in table_skip if b == "PC" and k != "overall" and v != "NA"}
    
    # Extract per-phase accuracies for macro-averaging (include approach)
    ic_accuracies_include = {k: float(v.rstrip('%')) for b, k, v in table_include if b == "IC" and k != "overall" and v != "NA"}
    pc_accuracies_include = {k: float(v.rstrip('%')) for b, k, v in table_include if b == "PC" and k != "overall" and v != "NA"}
    
    macro_acc_ic_skip = sum(ic_accuracies_skip.values()) / len(ic_accuracies_skip) if ic_accuracies_skip else None
    macro_acc_pc_skip = sum(pc_accuracies_skip.values()) / len(pc_accuracies_skip) if pc_accuracies_skip else None
    macro_acc_ic_include = sum(ic_accuracies_include.values()) / len(ic_accuracies_include) if ic_accuracies_include else None
    macro_acc_pc_include = sum(pc_accuracies_include.values()) / len(pc_accuracies_include) if pc_accuracies_include else None
    
    # Print skip approach metrics
    print(f"\nSkip Gold NA Approach:")
    print(f"Macro-avg F1 Score     IC {macro_f1_ic_skip*100:5.1f}%" if macro_f1_ic_skip is not None else "Macro-avg F1 Score     IC   N/A")
    print(f"Macro-avg F1 Score     PC {macro_f1_pc_skip*100:5.1f}%" if macro_f1_pc_skip is not None else "Macro-avg F1 Score     PC   N/A")
    print(f"Macro-avg Accuracy     IC {macro_acc_ic_skip:5.1f}%" if macro_acc_ic_skip is not None else "Macro-avg Accuracy     IC   N/A")
    print(f"Macro-avg Accuracy     PC {macro_acc_pc_skip:5.1f}%" if macro_acc_pc_skip is not None else "Macro-avg Accuracy     PC   N/A")
    
    # Print include approach metrics (accuracy only)
    print(f"\nInclude Gold NA Approach:")
    print(f"Macro-avg Accuracy     IC {macro_acc_ic_include:5.1f}%" if macro_acc_ic_include is not None else "Macro-avg Accuracy     IC   N/A")
    print(f"Macro-avg Accuracy     PC {macro_acc_pc_include:5.1f}%" if macro_acc_pc_include is not None else "Macro-avg Accuracy     PC   N/A")
    print()

    # ----- save -----
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    json.dump({
        # Skip Gold NA Approach
        "skip_gold_na": {
            "per_phase_accuracy": {b+"_"+k: v for b,k, v in table_skip},
            "mean_hit_rate_ic":  m_ic_skip, "stdev_hit_rate_ic": s_ic_skip,
            "mean_hit_rate_pc":  m_pc_skip, "stdev_hit_rate_pc": s_pc_skip,
            "per_phase_f1_ic": {k: f"{v*100:.1f}%" if v is not None else "N/A" for k, v in ic_f1_scores_skip.items()},
            "per_phase_f1_pc": {k: f"{v*100:.1f}%" if v is not None else "N/A" for k, v in pc_f1_scores_skip.items()},
            "macro_avg_f1_ic": f"{macro_f1_ic_skip*100:.1f}%" if macro_f1_ic_skip is not None else "N/A",
            "macro_avg_f1_pc": f"{macro_f1_pc_skip*100:.1f}%" if macro_f1_pc_skip is not None else "N/A",
            "macro_avg_accuracy_ic": f"{macro_acc_ic_skip:.1f}%" if macro_acc_ic_skip is not None else "N/A",
            "macro_avg_accuracy_pc": f"{macro_acc_pc_skip:.1f}%" if macro_acc_pc_skip is not None else "N/A"
        },
        
        # Include Gold NA Approach (accuracy only)
        "include_gold_na": {
            "per_phase_accuracy": {b+"_"+k: v for b,k, v in table_include},
            "mean_hit_rate_ic":  m_ic_include, "stdev_hit_rate_ic": s_ic_include,
            "mean_hit_rate_pc":  m_pc_include, "stdev_hit_rate_pc": s_pc_include,
            "macro_avg_accuracy_ic": f"{macro_acc_ic_include:.1f}%" if macro_acc_ic_include is not None else "N/A",
            "macro_avg_accuracy_pc": f"{macro_acc_pc_include:.1f}%" if macro_acc_pc_include is not None else "N/A"
        }
    }, open(sum_out,"w"), indent=2)

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

# ---------------------------------------------------------------------------
#  ENTRY-POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 Evaluation: Information & Procedural Compliance")
    parser.add_argument("--data-source", choices=["real", "synthetic"],
                       default="real", help="Data source to evaluate")
    parser.add_argument("--sample-limit", type=int, default=5,
                       help="Limit number of samples per generating model (default: 5)")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True,
                       help="Run API calls in parallel (default: True)")

    args = parser.parse_args()

    print(f"🚀 Starting Task 2 evaluation with data source: {args.data_source}")
    print(f"📊 Sample limit: {args.sample_limit}")
    print(f"⚙️  Execution mode: {'parallel' if args.parallel else 'sequential'}")

    # Run async version
    asyncio.run(run_all_models(args.data_source, args.sample_limit, args.parallel))
