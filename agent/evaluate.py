"""Build and evaluate one KernelForge kernel candidate.

    python agent/evaluate.py kernels/matmul/baseline.cu [M N K]

Always prints ONE JSON object and never raises on a bad candidate. Every
outcome the agent can cause maps to a `status`:

    ok             compiled, correct on all shapes, benchmarked
    compile_error  nvcc failed            -> `message` holds the compiler output
    incorrect      wrong results          -> `correctness.cases` shows every shape
    runtime_error  CUDA/cuBLAS error      -> `stage` + `message` (e.g. illegal address)
    timeout        hung (e.g. deadlocked __syncthreads) or compile took too long
    crash          process died without emitting JSON (segfault, abort)
    setup_error    nvcc missing, file missing, ...

Each candidate runs in its own process, so a poisoned CUDA context (illegal
memory access) can never leak into the next candidate.

Environment:
    KERNELFORGE_ARCH   nvcc -arch value. Default "native" (the GPU in this
                       machine). Set e.g. "sm_90" when building for another GPU.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "benchmark" / "runner.cu"

COMPILE_TIMEOUT_S = 180
RUN_TIMEOUT_S = 120


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def _head(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


def evaluate(
    kernel_path,
    shape=None,
    compile_timeout=COMPILE_TIMEOUT_S,
    run_timeout=RUN_TIMEOUT_S,
    workdir=None,
):
    kernel = Path(kernel_path).resolve()
    if not kernel.is_file():
        return {"status": "setup_error", "message": f"kernel file not found: {kernel}"}

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {"status": "setup_error", "message": "nvcc not found on PATH"}

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="kernelforge_"))
    work.mkdir(parents=True, exist_ok=True)
    exe = work / ("candidate.exe" if os.name == "nt" else "candidate")

    arch = os.environ.get("KERNELFORGE_ARCH", "native")
    compile_cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        "-lineinfo",  # lets Nsight Compute map metrics back to source lines
        str(RUNNER),
        str(kernel),
        "-o",
        str(exe),
        "-lcublas",
    ]

    # ---- compile ----
    try:
        built = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=compile_timeout
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stage": "compile", "limit_s": compile_timeout}

    if built.returncode != 0:
        # Compiler errors are at the top of nvcc output; keep the head.
        return {
            "status": "compile_error",
            "message": _head(built.stderr or built.stdout),
        }

    # ---- run ----
    run_cmd = [str(exe)] + [str(int(d)) for d in (shape or [])]
    try:
        ran = subprocess.run(
            run_cmd, capture_output=True, text=True, timeout=run_timeout
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stage": "run", "limit_s": run_timeout}

    lines = [ln for ln in ran.stdout.splitlines() if ln.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return {
            "status": "crash",
            "returncode": ran.returncode,
            "stderr": _tail(ran.stderr),
        }

    result["returncode"] = ran.returncode
    result["log"] = _tail(ran.stderr, 2000)
    result["binary"] = str(exe)  # for profiling (ncu) of the same build
    return result


def main():
    description = (__doc__ or "").split("\n")[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("kernel", help="path to the candidate .cu file")
    parser.add_argument("shape", nargs="*", type=int, help="optional: M N K")
    parser.add_argument("--run-timeout", type=int, default=RUN_TIMEOUT_S)
    parser.add_argument("--compile-timeout", type=int, default=COMPILE_TIMEOUT_S)
    args = parser.parse_args()

    if args.shape and len(args.shape) != 3:
        parser.error("shape must be exactly three integers: M N K")

    result = evaluate(
        args.kernel,
        shape=args.shape or None,
        compile_timeout=args.compile_timeout,
        run_timeout=args.run_timeout,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
