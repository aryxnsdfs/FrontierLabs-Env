"""
FrontierLabs-Env: FastAPI Server (main.py)
OpenEnv-compliant endpoints + HTML Mission Control Dashboard.
"""

import os
import json
import asyncio
import threading
import subprocess
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment import FrontierLabsEnv, TASKS
from graders import grade

# ---------------------------------------------------------------------------
# Pydantic models (typed per OpenEnv spec)
# ---------------------------------------------------------------------------

class ActionModel(BaseModel):
    action_type: str = Field(..., description="One of: write_file, run_script, submit")
    filename: Optional[str] = Field(None, description="Target filename")
    content: Optional[str] = Field(None, description="File content (for write_file)")

class ResetModel(BaseModel):
    task_id: Optional[str] = Field(None, description="Task to reset to. Defaults to task1_security_audit")

class ObservationModel(BaseModel):
    task_id: str
    step: int
    done: bool
    message: str
    files: Dict[str, str]
    metrics: Dict[str, Any]
    partial_score: float

class RewardModel(BaseModel):
    value: float
    explanation: str

class StepResponseModel(BaseModel):
    observation: ObservationModel
    reward: RewardModel
    done: bool
    info: Dict[str, Any]

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FrontierLabs-Env",
    description="OpenEnv-compliant AI Infrastructure simulation environment.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (thread-safe via asyncio lock)
_env = FrontierLabsEnv()
_env_lock = asyncio.Lock()
_baseline_results: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Platform Setup | Environment Settings</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #131314; /* Default dark background like Gemini */
    --surface: #1e1f20;
    --surface-hover: #282a2c;
    --border: #444746;
    --border-hover: #5f6368;
    --text-primary: #e3e3e3;
    --text-secondary: #c4c7c5;
    --accent: #a8c7fa;
    --green: #81c995;
    --yellow: #fde293;
    --red: #f28b82;
  }
  
  * { margin:0; padding:0; box-sizing:border-box; }
  
  body { 
    background-color: var(--bg); 
    color: var(--text-primary); 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
    line-height: 1.6;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  .container { 
    max-width: 1000px; 
    margin: 0 auto; 
    padding: 4rem 2rem; 
  }

  /* Header */
  header { 
    margin-bottom: 3.5rem; 
  }
  header h1 { 
    font-size: 2.25rem; 
    font-weight: 500; 
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.75rem;
  }
  header p { 
    color: var(--text-secondary); 
    font-size: 1.1rem; 
    max-width: 650px; 
  }

  /* Status bar */
  .status-bar { 
    display: flex; 
    gap: 0.75rem; 
    margin-bottom: 3rem; 
    flex-wrap: wrap; 
  }
  .status-chip { 
    border: 1px solid var(--border); 
    border-radius: 8px;
    padding: 0.5rem 0.875rem; 
    font-size: 0.875rem; 
    display: flex; 
    align-items: center; 
    gap: 0.5rem; 
    color: var(--text-secondary);
    background: transparent;
  }
  .status-chip .dot { 
    width: 6px; 
    height: 6px; 
    border-radius: 50%; 
    background: var(--green);
  }

  /* Cards grid */
  .grid { 
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); 
    gap: 1.25rem; 
    margin-bottom: 4rem; 
  }

  .card { 
    background: var(--surface); 
    border: 1px solid var(--border); 
    border-radius: 12px;
    padding: 1.5rem; 
    transition: background-color 0.2s ease, border-color 0.2s ease;
  }
  .card:hover { 
    background: var(--surface-hover);
    border-color: var(--border-hover);
  }

  .card h3 { 
    font-size: 1.1rem; 
    font-weight: 500; 
    color: var(--text-primary); 
    margin-bottom: 1.25rem;
  }

  /* Task specific */
  .task-header { 
    display: flex; 
    justify-content: space-between; 
    align-items: flex-start; 
    margin-bottom: 0.875rem; 
  }
  .task-name { 
    font-weight: 500; 
    font-size: 1rem; 
    color: #fff; 
  }
  .difficulty { 
    font-size: 0.75rem; 
    padding: 0.2rem 0.6rem; 
    border-radius: 12px; 
    font-weight: 500; 
  }
  .easy { background: rgba(129, 201, 149, 0.1); color: var(--green); }
  .medium { background: rgba(253, 226, 147, 0.1); color: var(--yellow); }
  .hard { background: rgba(242, 139, 130, 0.1); color: var(--red); }
  
  .task-desc { 
    font-size: 0.9rem; 
    color: var(--text-secondary); 
  }

  /* Endpoint list */
  .endpoint { 
    padding: 0.6rem 0; 
    border-bottom: 1px solid var(--border);
    font-family: 'Roboto Mono', monospace; 
    font-size: 0.85rem; 
    display: flex; 
    align-items: center; 
    gap: 1rem; 
  }
  .endpoint:last-child { border-bottom: none; padding-bottom: 0; }
  .endpoint:first-of-type { padding-top: 0; }
  .method { font-weight: 500; font-size: 0.75rem; width: 35px; }
  .get { color: var(--green); }
  .post { color: var(--accent); }
  .path { color: var(--text-primary); flex-grow: 1; }
  .ep-desc { color: var(--text-secondary); font-family: 'Inter', sans-serif; font-size: 0.85rem; text-align: right; }

  /* Metrics summary */
  .metric { 
    display: flex; 
    justify-content: space-between; 
    align-items: center;
    padding: 0.6rem 0; 
    border-bottom: 1px solid var(--border); 
  }
  .metric:last-child { border-bottom: none; padding-bottom: 0; }
  .metric:first-of-type { padding-top: 0; }
  .metric-label { color: var(--text-secondary); font-size: 0.9rem; }
  .metric-value { font-weight: 500; color: var(--text-primary); font-size: 0.9rem; }

  /* Code blocks */
  .code-block { 
    background: #111111; 
    border-radius: 8px;
    padding: 1.25rem; 
    font-family: 'Roboto Mono', monospace; 
    font-size: 0.85rem;
    line-height: 1.5; 
    overflow-x: auto; 
    color: var(--text-secondary); 
  }
  .code-comment { color: #8ab4f8; }

  a { color: var(--accent); text-decoration: none; transition: color 0.15s; }
  a:hover { color: #d3e3fd; text-decoration: underline; }

  footer { 
    color: var(--text-secondary); 
    font-size: 0.875rem; 
    padding: 2rem 0;
    border-top: 1px solid var(--border); 
    display: flex;
    justify-content: space-between;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Platform Setup</h1>
    <p>A structured environment for configuring and validating engineering infrastructure tasks.</p>
  </header>

  <div class="status-bar">
    <div class="status-chip"><span class="dot"></span> System Online</div>
    <div class="status-chip">3 Scenarios Available</div>
    <div class="status-chip">Standardized Validation</div>
    <div class="status-chip">REST API</div>
    <div class="status-chip">Containerized Environment</div>
  </div>

  <div class="grid">
    <!-- Task 1 -->
    <div class="card">
      <div class="task-header">
        <div class="task-name">Security Protocol Audit</div>
        <span class="difficulty easy">Tier 1</span>
      </div>
      <div class="task-desc">
        A dataset contains hidden vulnerabilities. Develop a script to scan and neutralize these threats, generating a clean output and a summary report of your findings.
      </div>
    </div>

    <!-- Task 2 -->
    <div class="card">
      <div class="task-header">
        <div class="task-name">Cluster Memory Reallocation</div>
        <span class="difficulty medium">Tier 2</span>
      </div>
      <div class="task-desc">
        The cluster is frequently failing due to memory exhaustion during high load periods. Refactor the underlying logic to distribute the load efficiently across the available system resources.
      </div>
    </div>

    <!-- Task 3 -->
    <div class="card">
      <div class="task-header">
        <div class="task-name">Hardware Bottleneck Resolution</div>
        <span class="difficulty hard">Tier 3</span>
      </div>
      <div class="task-desc">
        System latency is unusually high due to inefficient internal memory operations. Implement an optimized, lower-level set of instructions to fuse operations and significantly reduce the processing time per cycle.
      </div>
    </div>

    <!-- Quick API Overview -->
    <div class="card">
      <h3>API Reference</h3>
      <div class="endpoint"><span class="method post">POST</span><span class="path">/reset</span><span class="ep-desc">Initialize session</span></div>
      <div class="endpoint"><span class="method post">POST</span><span class="path">/step</span><span class="ep-desc">Submit step action</span></div>
      <div class="endpoint"><span class="method get">GET</span><span class="path">/state</span><span class="ep-desc">Retrieve internal state</span></div>
      <div class="endpoint"><span class="method get">GET</span><span class="path">/tasks</span><span class="ep-desc">List task parameters</span></div>
      <div class="endpoint"><span class="method get">GET</span><span class="path">/grader</span><span class="ep-desc">Evaluate score</span></div>
      <div class="endpoint"><span class="method get">GET</span><span class="path">/docs</span><span class="ep-desc">View specification</span></div>
    </div>

    <!-- System Stats -->
    <div class="card">
      <h3>System Overview</h3>
      <div class="metric"><span class="metric-label">Active Scenarios</span><span class="metric-value">3</span></div>
      <div class="metric"><span class="metric-label">Evaluation Metric</span><span class="metric-value">0.0 – 1.0</span></div>
      <div class="metric"><span class="metric-label">Cycle Limits (T1/T2/T3)</span><span class="metric-value">20/25/30</span></div>
      <div class="metric"><span class="metric-label">Operation Types</span><span class="metric-value">Read, Write, Execute</span></div>
      <div class="metric"><span class="metric-label">Architecture Core</span><span class="metric-value">FastAPI</span></div>
    </div>

    <!-- Usage Example -->
    <div class="card">
      <h3>Usage Example</h3>
      <div class="code-block">
<span class="code-comment"># Initialize a new session</span>
curl -X POST /reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task1_security_audit"}'

<span class="code-comment"># Submit a script modification</span>
curl -X POST /step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"write_file",
     "filename":"audit.py",
     "content":"..."}'
      </div>
    </div>
  </div>

  <footer>
    <span>Platform Environment v1.0.0</span>
    <span><a href="/docs">View API Docs</a></span>
  </footer>
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Mission Control Dashboard — human-readable environment overview."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.post("/reset", tags=["OpenEnv"])
async def reset(body: ResetModel = ResetModel()):
    """Reset the environment and return the initial observation."""
    async with _env_lock:
        obs = _env.reset(task_id=body.task_id)
    return {"observation": obs}


@app.post("/step", tags=["OpenEnv"])
async def step(action: ActionModel):
    """Take one action in the environment. Returns observation, reward, done, info."""
    async with _env_lock:
        obs, reward_value, done, info = _env.step(action.model_dump(exclude_none=True))
    return {
        "observation": obs,
        "reward": {"value": round(reward_value, 4), "explanation": info.get("explanation", "")},
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["OpenEnv"])
async def state():
    """Return full internal environment state (for debugging / judges)."""
    async with _env_lock:
        s = _env.state()
    return s


@app.get("/tasks", tags=["OpenEnv"])
async def tasks():
    """Return list of available tasks and the action schema."""
    return {
        "tasks": [
            {
                "id": "task1_security_audit",
                "name": "Security Audit & Self-Evaluation",
                "difficulty": "easy",
                "max_steps": 20,
                "success_threshold": 0.8,
                "description": "Detect and remove 50 backdoor prompts from dataset.jsonl. Write audit.py and evaluate.py, run them, then submit.",
            },
            {
                "id": "task2_fsdp_cluster",
                "name": "Distributed Cluster Crash (FSDP)",
                "difficulty": "medium",
                "max_steps": 25,
                "success_threshold": 0.8,
                "description": "Fix OOM crash in train.py by rewriting it as train_fsdp.py using PyTorch FSDP across 8 GPUs.",
            },
            {
                "id": "task3_triton_kernel",
                "name": "Triton Hardware Bottleneck",
                "difficulty": "hard",
                "max_steps": 30,
                "success_threshold": 0.8,
                "description": "Write a fused Triton kernel (fast_silu_kernel.py) replacing slow SiLU+multiply ops, targeting <20ms/step.",
            },
        ],
        "action_schema": {
            "action_type": {
                "type": "string",
                "enum": ["write_file", "run_script", "submit"],
                "required": True,
            },
            "filename": {
                "type": "string",
                "required": "for write_file and run_script",
            },
            "content": {
                "type": "string",
                "required": "for write_file",
            },
        },
    }


@app.get("/grader", tags=["OpenEnv"])
async def grader():
    """Grade the current episode and return score in [0.0, 1.0]."""
    async with _env_lock:
        s = _env.state()
        task_id = s["task_id"]
        get_file = _env.get_filesystem_file
        result = grade(task_id, s, get_file)
    return result


@app.post("/baseline", tags=["OpenEnv"])
async def baseline_endpoint():
    """
    Trigger the baseline inference script (non-blocking).
    Returns cached results if already computed.
    """
    if _baseline_results:
        return {"status": "cached", "results": _baseline_results}

    # Run baseline in background thread
    def run_baseline():
        try:
            result = subprocess.run(
                ["python", "inference.py", "--json-output"],
                capture_output=True, text=True, timeout=300
            )
            output = result.stdout.strip()
            if output:
                try:
                    data = json.loads(output.split("\n")[-1])
                    _baseline_results.update(data)
                except json.JSONDecodeError:
                    _baseline_results["error"] = "Could not parse baseline output"
                    _baseline_results["raw"] = output[-500:]
            else:
                _baseline_results["error"] = result.stderr[-500:] if result.stderr else "No output"
        except subprocess.TimeoutExpired:
            _baseline_results["error"] = "Baseline timed out after 300s"
        except Exception as e:
            _baseline_results["error"] = str(e)

    thread = threading.Thread(target=run_baseline, daemon=True)
    thread.start()

    return {
        "status": "running",
        "message": "Baseline agent started. Check /baseline again in ~60 seconds for results.",
    }


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "env": "FrontierLabs-Env", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
