import json
from graders import grade, grade_task1, grade_task2, grade_task3
from environment import FrontierLabsEnv

env = FrontierLabsEnv()

for task in ["task1_security_audit", "task2_fsdp_cluster", "task3_triton_kernel"]:
    env.reset(task)
    # Test without creating any files
    res = grade(task, env.state(), env.get_filesystem_file)
    print(f"{task} empty: score = {res['score']}")

    # Let's write the golden solutions and see
    if task == "task1_security_audit":
        # Simulate perfect solution
        cleaned = []
        for l in env._filesystem["dataset.jsonl"].strip().split("\n"):
            p = json.loads(l)
            if p["label"] == "clean":
                cleaned.append(p)
        env._filesystem["cleaned_dataset.jsonl"] = "\n".join(json.dumps(p) for p in cleaned)
        report = {"f1_score": 1.0}
        env._filesystem["metrics_report.json"] = json.dumps(report)
        res = grade(task, env.state(), env.get_filesystem_file)
        print(f"{task} perfect: score = {res['score']}")
    elif task == "task2_fsdp_cluster":
        # Perfect
        env._filesystem["train_fsdp.py"] = """
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
dist.init_process_group("nccl")
        """
        res = grade(task, env.state(), env.get_filesystem_file)
        print(f"{task} perfect: score = {res['score']}")
    elif task == "task3_triton_kernel":
        env._filesystem["fast_silu_kernel.py"] = """
import triton
import triton.language as tl
import torch
@triton.jit
def kernel(x_ptr, gate_ptr):
    x = tl.load(x_ptr)
    gate = tl.load(gate_ptr)
    y = x * sigmoid(x) * gate
    tl.store(gate_ptr, y)
        """
        res = grade(task, env.state(), env.get_filesystem_file)
        print(f"{task} perfect: score = {res['score']}")
