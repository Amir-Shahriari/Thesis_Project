"""Write the execution stack of a run to a JSON file beside its artefacts.

Exists because the 50x50 action-space A/B was nearly invalidated by an
undetected stack change: `.venv` keeps sys.prefix pointed at itself, so a
`uv sync` silently replaced a CUDA torch with the +cpu wheel and every
subsequent run went to CPU without recording it. No artefact carried a device
or torch version, so which runs were GPU could not be established afterwards.
Every sweep should drop one of these next to its output.

Usage:  python scripts/analysis/record_provenance.py <out_dir> [label]
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out_dir = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    dev = "cpu"
    name = None
    cap = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        name = torch.cuda.get_device_name(0)
        cap = list(torch.cuda.get_device_capability(0))

    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        rev = None

    rec = {
        "label": label,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "cuda_available": torch.cuda.is_available(),
        "resolved_device": dev,
        "gpu_name": name,
        "compute_capability": cap,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "python": platform.python_version(),
        "git_rev": rev,
        "env": {k: os.environ.get(k) for k in (
            "QWARM_MASK_VISITED", "QWARM_SEEDS", "QWARM_OUT_DIR",
            "QWARM_SKIP_4X", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES")},
    }

    path = out_dir / "provenance.json"
    path.write_text(json.dumps(rec, indent=2))
    print("provenance -> %s" % path)
    print("  torch %s  device %s  %s" % (rec["torch_version"], dev, name or ""))


if __name__ == "__main__":
    main()
