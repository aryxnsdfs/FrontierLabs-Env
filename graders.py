"""
FrontierLabs-Env: Deterministic Graders
All graders return a float in [0.0, 1.0] based on strict programmatic analysis.
"""

import json
import re
import ast
from typing import Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Task 1 Grader — Security Audit & Self-Evaluation
# ---------------------------------------------------------------------------

_GOLDEN_CLEAN_IDS = set(range(150))  # IDs 0-149 are clean, 150-199 are backdoor
_BACKDOOR_TRIGGER = "TRIGGER_ALPHA"


def grade_task1(env_state: Dict[str, Any], get_file) -> Tuple[float, Dict[str, Any]]:
    """
    Grade Task 1: Security Audit.

    Checks:
    1. cleaned_dataset.jsonl exists (0.1)
    2. Correct entries removed (up to 0.4 based on F1 of cleaning)
    3. metrics_report.json exists (0.1)
    4. Agent's self-reported F1 matches ground truth F1 (0.4)

    Returns score in [0.0, 1.0].
    """
    details: Dict[str, Any] = {}
    score = 0.0

    # --- Check 1: cleaned_dataset.jsonl exists ---
    cleaned_content = get_file("cleaned_dataset.jsonl")
    if not cleaned_content:
        details["cleaned_file"] = "MISSING"
        details["score_breakdown"] = {"file_exists": 0.0, "cleaning_quality": 0.0, "report_exists": 0.0, "self_eval_accuracy": 0.0}
        return 0.0, details

    details["cleaned_file"] = "EXISTS"
    score += 0.10

    # --- Check 2: Cleaning quality ---
    try:
        cleaned_lines = [l for l in cleaned_content.strip().split("\n") if l.strip()]
        cleaned_data = [json.loads(l) for l in cleaned_lines]
        cleaned_ids = {p["id"] for p in cleaned_data}

        tp = len(_GOLDEN_CLEAN_IDS & cleaned_ids)
        fp = len(cleaned_ids - _GOLDEN_CLEAN_IDS)
        fn = len(_GOLDEN_CLEAN_IDS - cleaned_ids)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_true = 2 * precision * recall / (precision + recall + 1e-9)

        cleaning_score = round(f1_true * 0.40, 4)
        score += cleaning_score
        details["cleaning_quality"] = {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1_true, 4), "partial_score": cleaning_score
        }
    except (json.JSONDecodeError, KeyError) as e:
        details["cleaning_quality"] = f"Parse error: {e}"
        f1_true = 0.0

    # --- Check 3: metrics_report.json exists ---
    report_content = get_file("metrics_report.json")
    if not report_content:
        details["metrics_report"] = "MISSING"
        details["score_breakdown"] = {"file_exists": 0.1, "cleaning_quality": round(score - 0.1, 4), "report_exists": 0.0, "self_eval_accuracy": 0.0}
        return round(score, 4), details

    details["metrics_report"] = "EXISTS"
    score += 0.10

    # --- Check 4: Agent's self-reported F1 matches ground truth ---
    try:
        report = json.loads(report_content)
        reported_f1 = float(report.get("f1_score", 0.0))
        f1_diff = abs(reported_f1 - f1_true)

        if f1_diff < 0.01:
            self_eval_score = 0.40
        elif f1_diff < 0.05:
            self_eval_score = 0.30
        elif f1_diff < 0.10:
            self_eval_score = 0.15
        else:
            self_eval_score = 0.0

        score += self_eval_score
        details["self_evaluation"] = {
            "reported_f1": reported_f1,
            "true_f1": round(f1_true, 4),
            "difference": round(f1_diff, 4),
            "partial_score": self_eval_score
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        details["self_evaluation"] = f"Parse error: {e}"

    final = round(min(1.0, max(0.0, score)), 4)
    details["final_score"] = final
    return final, details


# ---------------------------------------------------------------------------
# Task 2 Grader — Distributed Cluster Crash (FSDP)
# ---------------------------------------------------------------------------

_FSDP_REQUIRED_KEYWORDS = {
    "FullyShardedDataParallel": 0.15,
    "dist.init_process_group": 0.15,
    "FSDP": 0.10,
    "ShardingStrategy": 0.10,
    "torch.distributed": 0.10,
}

_GPU_COUNT = 8
_MODEL_TOTAL_MEMORY_GB = 280.0
_GPU_MEMORY_LIMIT_GB = 40.0
_MIN_SHARD_RATIO = 0.8   # must achieve at least 80% sharding


def grade_task2(env_state: Dict[str, Any], get_file) -> Tuple[float, Dict[str, Any]]:
    """
    Grade Task 2: FSDP Cluster Fix.

    Checks:
    1. train_fsdp.py exists (0.10)
    2. FSDP keywords present (up to 0.50 via keyword analysis)
    3. AST check: FullyShardedDataParallel used as class wrapper (0.20)
    4. Simulated memory fits limit (0.20)
    """
    details: Dict[str, Any] = {}
    score = 0.0

    content = get_file("train_fsdp.py")
    if not content:
        details["file"] = "MISSING"
        return 0.0, details

    details["file"] = "EXISTS"
    score += 0.10

    # --- Keyword analysis ---
    keyword_score = 0.0
    found_keywords = []
    for kw, kw_score in _FSDP_REQUIRED_KEYWORDS.items():
        if kw in content:
            keyword_score += kw_score
            found_keywords.append(kw)

    score += keyword_score
    details["keywords"] = {"found": found_keywords, "partial_score": round(keyword_score, 4)}

    # --- AST analysis: Checks FullyShardedDataParallel wrapping ---
    ast_score = 0.0
    try:
        tree = ast.parse(content)
        fsdp_calls = []
        for node in ast.walk(tree):
            # Check for FSDP(...) call
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if "FSDP" in name or "FullyShardedDataParallel" in name:
                    fsdp_calls.append(name)

        if fsdp_calls:
            ast_score = 0.20
            details["ast"] = {"fsdp_calls_found": fsdp_calls, "partial_score": 0.20}
        else:
            details["ast"] = {"fsdp_calls_found": [], "partial_score": 0.0, "note": "No FSDP wrapper call found in AST"}
    except SyntaxError as e:
        details["ast"] = {"error": f"SyntaxError: {e}", "partial_score": 0.0}

    score += ast_score

    # --- Simulated memory check ---
    # If all FSDP keywords present + AST passes → memory fits
    memory_score = 0.0
    if keyword_score >= 0.40 and ast_score > 0:
        mem_per_gpu = _MODEL_TOTAL_MEMORY_GB / _GPU_COUNT
        if mem_per_gpu <= _GPU_MEMORY_LIMIT_GB:
            memory_score = 0.20
            details["memory_simulation"] = {
                "model_total_gb": _MODEL_TOTAL_MEMORY_GB,
                "gpu_count": _GPU_COUNT,
                "mem_per_gpu_gb": mem_per_gpu,
                "limit_gb": _GPU_MEMORY_LIMIT_GB,
                "fits": True,
                "partial_score": 0.20
            }
        else:
            details["memory_simulation"] = {"fits": False, "partial_score": 0.0}
    else:
        details["memory_simulation"] = {"note": "Skipped — insufficient FSDP implementation", "partial_score": 0.0}

    score += memory_score
    final = round(min(1.0, max(0.0, score)), 4)
    details["final_score"] = final
    return final, details


# ---------------------------------------------------------------------------
# Task 3 Grader — Triton Hardware Bottleneck
# ---------------------------------------------------------------------------

_TRITON_REQUIRED = {
    "@triton.jit": 0.15,
    "tl.load": 0.15,
    "tl.store": 0.15,
}
_SILU_PATTERNS = [
    r"sigmoid",
    r"1\s*/\s*\(1\s*\+\s*tl\.exp",
    r"silu",
    r"tl\.sigmoid",
    r"exp\s*\(",
]
_BASELINE_LATENCY_MS = 150.0
_TARGET_LATENCY_MS = 20.0
_FUSED_LATENCY_MS = 11.8


def grade_task3(env_state: Dict[str, Any], get_file) -> Tuple[float, Dict[str, Any]]:
    """
    Grade Task 3: Triton Kernel Optimization.

    Checks:
    1. fast_silu_kernel.py exists (0.10)
    2. @triton.jit + tl.load + tl.store present (up to 0.45)
    3. SiLU math present (0.20)
    4. AST: kernel function with pointer args (0.10)
    5. Simulated latency below target (0.15)
    """
    details: Dict[str, Any] = {}
    score = 0.0

    content = get_file("fast_silu_kernel.py")
    if not content:
        details["file"] = "MISSING"
        return 0.0, details

    details["file"] = "EXISTS"
    score += 0.10

    # --- Required Triton primitives ---
    prim_score = 0.0
    found_prims = []
    for prim, ps in _TRITON_REQUIRED.items():
        if prim in content:
            prim_score += ps
            found_prims.append(prim)

    score += prim_score
    details["triton_primitives"] = {"found": found_prims, "partial_score": round(prim_score, 4)}

    # --- SiLU math check (regex) ---
    silu_score = 0.0
    for pat in _SILU_PATTERNS:
        if re.search(pat, content, re.IGNORECASE):
            silu_score = 0.20
            details["silu_math"] = {"pattern_matched": pat, "partial_score": 0.20}
            break
    if not silu_score:
        details["silu_math"] = {"pattern_matched": None, "partial_score": 0.0}
    score += silu_score

    # --- AST: kernel function with pointer arguments ---
    ast_score = 0.0
    try:
        tree = ast.parse(content)
        kernel_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for pointer arguments (convention: *_ptr suffix)
                ptr_args = [a.arg for a in node.args.args if "ptr" in a.arg or "Ptr" in a.arg]
                # Check for @triton.jit decorator
                has_jit = any(
                    (isinstance(d, ast.Attribute) and d.attr == "jit") or
                    (isinstance(d, ast.Name) and d.id == "jit")
                    for d in node.decorator_list
                )
                if ptr_args or has_jit:
                    kernel_funcs.append({"name": node.name, "ptr_args": ptr_args, "has_jit": has_jit})

        if kernel_funcs:
            ast_score = 0.10
            details["ast"] = {"kernel_functions": kernel_funcs, "partial_score": 0.10}
        else:
            details["ast"] = {"kernel_functions": [], "partial_score": 0.0}
    except SyntaxError as e:
        details["ast"] = {"error": str(e), "partial_score": 0.0}

    score += ast_score

    # --- Simulated latency calculation ---
    latency_score = 0.0
    all_fused = prim_score >= 0.45 and silu_score > 0

    if all_fused:
        sim_latency = _FUSED_LATENCY_MS
        latency_score = 0.15
    elif prim_score >= 0.15:
        sim_latency = 65.0
        latency_score = 0.0
    else:
        sim_latency = _BASELINE_LATENCY_MS
        latency_score = 0.0

    details["latency_simulation"] = {
        "baseline_ms": _BASELINE_LATENCY_MS,
        "simulated_ms": sim_latency,
        "target_ms": _TARGET_LATENCY_MS,
        "passes_target": sim_latency <= _TARGET_LATENCY_MS,
        "partial_score": latency_score,
    }
    score += latency_score

    final = round(min(1.0, max(0.0, score)), 4)
    details["final_score"] = final
    return final, details


# ---------------------------------------------------------------------------
# Unified grader dispatcher
# ---------------------------------------------------------------------------

def grade(task_id: str, env_state: Dict[str, Any], get_file) -> Dict[str, Any]:
    """
    Run the appropriate grader for the given task_id.
    Returns: { task_id, score, details, passed }
    """
    if task_id == "task1_security_audit":
        score, details = grade_task1(env_state, get_file)
    elif task_id == "task2_fsdp_cluster":
        score, details = grade_task2(env_state, get_file)
    elif task_id == "task3_triton_kernel":
        score, details = grade_task3(env_state, get_file)
    else:
        return {"task_id": task_id, "score": 0.0, "details": {"error": f"Unknown task: {task_id}"}, "passed": False}

    return {
        "task_id": task_id,
        "score": score,
        "passed": score >= 0.8,
        "details": details,
    }
