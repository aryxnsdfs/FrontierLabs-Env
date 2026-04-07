"""
FrontierLabs-Env: Baseline Inference Script (inference.py)

Uses the OpenAI API to autonomously solve all 3 tasks in the FrontierLabs-Env.
Produces reproducible baseline scores for hackathon validation.

Usage:
    python inference.py                    # Pretty output
    python inference.py --json-output      # JSON output for /baseline endpoint
    python inference.py --task task1_security_audit  # Single task
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Dict, Any, Optional

try:
    import openai
except ImportError:
    print("OpenAI package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("FRONTIER_ENV_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS_PER_TASK = 15

# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grader() -> Dict[str, Any]:
    r = requests.get(f"{BASE_URL}/grader", timeout=30)
    r.raise_for_status()
    return r.json()

def env_state() -> Dict[str, Any]:
    r = requests.get(f"{BASE_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# System prompt for the agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI Infrastructure Engineer. You are operating inside a simulated
GPU supercomputing environment. You interact with the environment by outputting JSON actions.

Available actions:
1. {"action_type": "write_file", "filename": "<name>", "content": "<code>"}
2. {"action_type": "run_script", "filename": "<name>"}
3. {"action_type": "submit"}

Rules:
- Always output EXACTLY ONE JSON object per response, starting with { and ending with }.
- No markdown, no explanation, no backticks. ONLY the raw JSON action.
- Read the task in the observation message and files carefully before acting.
- For write_file, include complete, functional code in "content".
- Call run_script after writing each script.
- Call submit when you believe the task is complete.
"""

# ---------------------------------------------------------------------------
# Task-specific expert prompts (fallback if API key not set)
# ---------------------------------------------------------------------------

EXPERT_SOLUTIONS = {
    "task1_security_audit": [
        {
            "action_type": "write_file",
            "filename": "audit.py",
            "content": """import json

with open('dataset.jsonl', 'r') as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    entry = json.loads(line.strip())
    if 'TRIGGER_ALPHA' not in entry.get('prompt', ''):
        cleaned.append(entry)

with open('cleaned_dataset.jsonl', 'w') as f:
    for entry in cleaned:
        f.write(json.dumps(entry) + '\\n')

print(f'Removed {len(lines) - len(cleaned)} backdoor entries. Kept {len(cleaned)} clean entries.')
"""
        },
        {"action_type": "run_script", "filename": "audit.py"},
        {
            "action_type": "write_file",
            "filename": "evaluate.py",
            "content": """import json

with open('cleaned_dataset.jsonl', 'r') as f:
    cleaned_lines = f.readlines()
cleaned = [json.loads(l) for l in cleaned_lines]
cleaned_ids = set(p['id'] for p in cleaned)

with open('golden_baseline.jsonl', 'r') as f:
    golden_lines = f.readlines()
golden = [json.loads(l) for l in golden_lines]
golden_ids = set(p['id'] for p in golden)

tp = len(golden_ids & cleaned_ids)
fp = len(cleaned_ids - golden_ids)
fn = len(golden_ids - cleaned_ids)
tn = 50 - fp

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

report = {
    'true_positives': tp, 'true_negatives': tn,
    'false_positives': fp, 'false_negatives': fn,
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4)
}
with open('metrics_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print('metrics_report.json written:', report)
"""
        },
        {"action_type": "run_script", "filename": "evaluate.py"},
        {"action_type": "submit"},
    ],
    "task2_fsdp_cluster": [
        {
            "action_type": "write_file",
            "filename": "train_fsdp.py",
            "content": """import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

class MegaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(8192, 8192, bias=False) for _ in range(80)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train():
    # Initialize distributed process group across 8 GPUs
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    model = MegaModel()
    # Wrap with FSDP — shards model across all 8 GPUs
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(1000):
        x = torch.randn(4, 8192, device=f'cuda:{local_rank}')
        loss = model(x).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0 and local_rank == 0:
            print(f'Step {step}, loss: {loss.item():.4f}')

    dist.destroy_process_group()

if __name__ == '__main__':
    train()
"""
        },
        {"action_type": "run_script", "filename": "train_fsdp.py"},
        {"action_type": "submit"},
    ],
    "task3_triton_kernel": [
        {
            "action_type": "write_file",
            "filename": "fast_silu_kernel.py",
            "content": """import triton
import triton.language as tl
import torch

@triton.jit
def fused_silu_multiply_kernel(
    x_ptr, gate_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    \"\"\"Fused SiLU + element-wise multiply kernel.
    All ops happen in registers — single memory round-trip.
    \"\"\"
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x and gate from global memory (single read each)
    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)

    # Compute SiLU in registers: silu(x) = x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x

    # Fused multiply with gate (in registers)
    output = silu_x * gate

    # Single write to global memory
    tl.store(output_ptr + offsets, output, mask=mask)


def fast_silu_multiply(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    \"\"\"Drop-in replacement for slow_silu_multiply using the fused Triton kernel.\"\"\"
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_silu_multiply_kernel[grid](x, gate, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == '__main__':
    x = torch.randn(4096, device='cuda')
    gate = torch.randn(4096, device='cuda')
    out = fast_silu_multiply(x, gate)
    print(f'Output shape: {out.shape}, mean: {out.mean().item():.4f}')
    print('Kernel executed successfully.')
"""
        },
        {"action_type": "run_script", "filename": "fast_silu_kernel.py"},
        {"action_type": "submit"},
    ],
}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task_with_llm(client: openai.OpenAI, task_id: str, verbose: bool = True) -> float:
    """Run an LLM agent against the given task using the OpenAI client. Returns final grader score."""
    print(f"[START] task={task_id}", flush=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id}")
        print(f"{'='*60}")

    # Reset env
    reset_resp = env_reset(task_id)
    obs = reset_resp["observation"]

    # --- SAFELY GRAB THE README CONTENT ---
    files_dict = obs.get('files', {})
    file_keys = list(files_dict.keys())
    
    # Extract just the first part of the task ID (e.g., 'task1' from 'task1_security_audit')
    task_short = task_id.split("_")[0] 
    readme_key = f"README_{task_short}.txt"
    
    # Safely get the content. If the README doesn't exist, it defaults to an empty string
    readme_content = files_dict.get(readme_key, "")
    # --------------------------------------

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Task: {obs['message']}\n\n"
            f"Available files: {file_keys}\n\n"
            f"README:\n{readme_content}\n\n"
            "Begin solving. Output your first action as a JSON object."
        )}
    ]

    actual_steps = 0
    for step_num in range(MAX_STEPS_PER_TASK):
        actual_steps += 1
        if verbose:
            print(f"\n  Step {step_num + 1}/{MAX_STEPS_PER_TASK}:")

        # Call LLM with retry
        max_retries = 3
        raw = ""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000,
                )
                # FIX: Access the first choice in the choices list
                if response.choices and response.choices[0].message.content:
                    raw = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if verbose:
                    print(f"    API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"[END] task={task_id} score=0.0010 steps={actual_steps}", flush=True)
                    return 0.001
                time.sleep(3)
        
        if not raw:
            if verbose:
                print("    Failed to get valid text from API response. Skipping step.")
            messages.append({"role": "user", "content": "Your last response was empty or blocked. Please provide a valid JSON action."})
            continue

        if verbose:
            print(f"    Agent: {raw[:200]}")

        # Parse JSON action
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                action = json.loads(raw[start:end])
            else:
                if verbose:
                    print("    Could not parse JSON action. Skipping.")
                continue
        except json.JSONDecodeError as e:
            if verbose:
                print(f"    JSON parse error: {e}")
            continue

        # Send action to env
        try:
            step_resp = env_step(action)
        except Exception as e:
            if verbose:
                print(f"    Env step error: {e}")
            break

        obs = step_resp["observation"]
        reward = step_resp["reward"]["value"]
        done = step_resp["done"]
        expl = step_resp["reward"]["explanation"]
        
        print(f"[STEP] step={actual_steps} reward={reward:.4f}", flush=True)

        if verbose:
            print(f"    Reward: {reward:+.3f} | {expl[:100]}")
            print(f"    Partial score: {obs['partial_score']:.3f}")

        messages.append({"role": "assistant", "content": raw})
        
        warning = ""
        if reward < 0:
            warning = "WARNING: Your last action resulted in a negative reward or failure. DO NOT repeat the exact same action. Try a different approach.\n\n"

        messages.append({"role": "user", "content": (
            f"Step result (Step {step_num + 1}/{MAX_STEPS_PER_TASK}):\n"
            f"Reward: {reward:.4f}\n"
            f"Explanation: {expl}\n"
            f"Metrics: {json.dumps(obs['metrics'])}\n"
            f"Files now on filesystem: {list(obs['files'].keys())}\n\n"
            f"{warning}"
            f"{'Episode done.' if done else 'Continue solving. Output your next action as EXACTLY ONE JSON object.'}"
        )})

        if done:
            break

        time.sleep(0.2)

    grade_resp = env_grader()
    score = grade_resp["score"]
    
    print(f"[END] task={task_id} score={score:.4f} steps={actual_steps}", flush=True)
    
    if verbose:
        print(f"\n  Final grader score: {score:.4f}")
        print(f"  Passed: {grade_resp['passed']}")
    return score

def run_task_with_expert(task_id: str, verbose: bool = True) -> float:
    """Run the deterministic expert solution. Used when no API key is set."""
    print(f"[START] task={task_id}", flush=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id} (expert solution — no API key)")
        print(f"{'='*60}")

    env_reset(task_id)
    actions = EXPERT_SOLUTIONS.get(task_id, [])

    actual_steps = 0
    for i, action in enumerate(actions):
        actual_steps += 1
        if verbose:
            print(f"  Step {i+1}: {action['action_type']} {action.get('filename', '')}")
        
        resp = env_step(action)
        reward = resp["reward"]["value"]
        expl = resp["reward"]["explanation"]
        
        print(f"[STEP] step={actual_steps} reward={reward:.4f}", flush=True)
        
        if verbose:
            print(f"    Reward: {reward:+.3f} | {expl[:100]}")
        if resp["done"]:
            break
        time.sleep(0.1)

    grade_resp = env_grader()
    score = grade_resp["score"]
    
    print(f"[END] task={task_id} score={score:.4f} steps={actual_steps}", flush=True)
    
    if verbose:
        print(f"\n  Final grader score: {score:.4f} | Passed: {grade_resp['passed']}")
    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FrontierLabs-Env Baseline Agent")
    parser.add_argument("--task", type=str, default=None, help="Run single task by ID")
    parser.add_argument("--json-output", action="store_true", help="Output JSON for /baseline endpoint")
    parser.add_argument("--url", type=str, default=None, help="Override environment URL")
    args = parser.parse_args()

    global BASE_URL
    if args.url:
        BASE_URL = args.url

    tasks_to_run = (
        [args.task] if args.task
        else ["task1_security_audit", "task2_fsdp_cluster", "task3_triton_kernel"]
    )

    use_llm = bool(HF_TOKEN)
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if use_llm else None
    verbose = not args.json_output

    if verbose:
        print("FrontierLabs-Env Baseline Agent")
        print(f"Model: {MODEL_NAME if use_llm else 'expert (no HF_TOKEN)'}")
        print(f"Server: {BASE_URL}")

    results = {}
    for task_id in tasks_to_run:
        try:
            if use_llm and client:
                score = run_task_with_llm(client, task_id, verbose=verbose)
            else:
                score = run_task_with_expert(task_id, verbose=verbose)
            results[task_id] = {"score": round(score, 4), "passed": score >= 0.8}
        except Exception as e:
            import traceback
            print(f"\n[CRASH DETECTED ON {task_id}]")
            traceback.print_exc() 
            
            # 🔴 CHANGE: Set fallback score to 0.001 instead of 0.0
            print(f"[END] task={task_id} score=0.0010 steps=0", flush=True)
            results[task_id] = {"score": 0.001, "passed": False, "error": str(e)}
            if verbose:
                print(f"  ERROR on {task_id}: {e}")
    
    # 🔴 CHANGE: Set fallback average to 0.001
    avg = sum(r["score"] for r in results.values()) / len(results) if results else 0.001
    summary = {
        "model": MODEL_NAME if use_llm else "expert",
        "task_results": results,
        "average_score": round(avg, 4),
        "all_passed": all(r["passed"] for r in results.values()),
    }

    if verbose:
        print(f"\n{'='*60}")
        print("  BASELINE RESULTS SUMMARY")
        print(f"{'='*60}")
        for tid, res in results.items():
            status = "✅ PASS" if res["passed"] else "❌ FAIL"
            print(f"  {tid}: {res['score']:.4f} {status}")
        print(f"\n  Average Score: {avg:.4f}")
        print(f"  All Tasks Passed: {summary['all_passed']}")
    else:
        # JSON output for /baseline endpoint parsing
        print(json.dumps(summary))

    return summary


if __name__ == "__main__":
    main()