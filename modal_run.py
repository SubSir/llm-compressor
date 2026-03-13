import subprocess
from pathlib import Path
import modal

app = modal.App("llm-compressor-quant-eval")

base_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
).apt_install("git")

local_image = base_image.run_commands(
    "git clone https://github.com/SubSir/llm-compressor.git /root/llm-compressor",
    "cd /root/llm-compressor && pip install -e .",
    "pip install -U transformers",
    "pip install \"httpx[socks]\"",
    "pip install torchvision",
    "pip install lm_eval==0.4.11",
)


def _quantized_dir(model_name: str) -> str:
    model_suffix = model_name.rstrip("/").split("/")[-1]
    return f"{model_suffix}-awq-asym-fake"


@app.function(
    gpu="H200",
    timeout=7200,
    image=local_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cloud="aws",
)
def run_quant_and_eval(
    model_name: str = "Qwen/Qwen3.5-0.8B-Base",
    eval_seqlen: int = 8192,
    seed: int = 0,
) -> dict:
    repo_dir = Path("/root/llm-compressor")

    quant_cmd = [
        "python",
        "qwen3_5_awq.py",
        "--model-name",
        model_name,
    ]
    quant_proc = subprocess.run(
        quant_cmd,
        cwd=repo_dir,
        text=True,
        capture_output=True,
        check=True,
    )

    quant_dir = _quantized_dir(model_name)
    eval_cmd = [
        "python",
        "eval.py",
        "--model",
        quant_dir,
        "--seqlen",
        str(eval_seqlen),
        "--seed",
        str(seed),
    ]
    eval_proc = subprocess.run(
        eval_cmd,
        cwd=repo_dir,
        text=True,
        capture_output=True,
        check=True,
    )

    return {
        "quant_dir": quant_dir,
        "quant_stdout": quant_proc.stdout,
        "quant_stderr": quant_proc.stderr,
        "eval_stdout": eval_proc.stdout,
        "eval_stderr": eval_proc.stderr,
    }


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3.5-0.8B-Base",
    eval_seqlen: int = 8192,
    seed: int = 0,
):
    result = run_quant_and_eval.remote(
        model_name=model_name,
        eval_seqlen=eval_seqlen,
        seed=seed,
    )

    print("Quantized model dir:", result["quant_dir"])
    if result["quant_stdout"]:
        print("\n=== Quantization stdout ===\n")
        print(result["quant_stdout"])
    if result["quant_stderr"]:
        print("\n=== Quantization stderr ===\n")
        print(result["quant_stderr"])

    if result["eval_stdout"]:
        print("\n=== Evaluation stdout ===\n")
        print(result["eval_stdout"])
    if result["eval_stderr"]:
        print("\n=== Evaluation stderr ===\n")
        print(result["eval_stderr"])
