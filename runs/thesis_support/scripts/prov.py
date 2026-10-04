"""Shared provenance + run helpers for the thesis_support regeneration."""
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

REPO = pathlib.Path(r"C:\Users\amirh\Desktop\Thesis_Project")
OUT_DIR = REPO / "runs" / "thesis_support"
SCRATCH = pathlib.Path(__file__).parent
PY = sys.executable


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True).stdout.strip()


def environment():
    import numpy, scipy
    env = {"python": sys.version, "python_executable": PY,
           "platform": platform.platform(), "numpy": numpy.__version__,
           "scipy": scipy.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_name"] = (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None)
    except Exception as e:  # pragma: no cover
        env["torch"] = f"unavailable: {e}"
    env["qwarm_import_resolves_to"] = subprocess.run(
        [PY, "-c", "import sys; sys.path.insert(0, r'" + str(REPO / "src") +
         "'); import qwarm.env.dynamic_graph as m; print(m.__file__)"],
        capture_output=True, text=True).stdout.strip()
    env["device_used"] = "cpu"
    env["device_note"] = ("Deterministic graph analysis (numpy/scipy/pure "
                          "Python). No checkpoint or torch model is loaded, so "
                          "the GPU is not used.")
    env["interpreter_note"] = (
        "Thesis_Project has no .venv (C:/Users/amirh/Desktop/Thesis_Project/"
        ".venv/Scripts/python.exe does not exist). The analysis scripts "
        "hard-code sys.path to C:/Users/amirh/Desktop/qwarm-gnn-rl/src, so the "
        "interpreter of that tree's venv was used ONLY as a Python runtime "
        "(numpy/scipy). All qwarm imports resolve to Thesis_Project/src: the "
        "scripts were run from verbatim scratch copies whose only change is "
        "that sys.path line. src/qwarm/env is byte-identical between the two "
        "trees (diff -r, excluding __pycache__).")
    return env


def provenance(original_scripts, commands, scratch_copies=None, extra=None):
    rev = git(REPO, "rev-parse", "HEAD")
    status = git(REPO, "status", "--porcelain")
    d = {
        "git_repo": str(REPO),
        "git_revision": rev,
        "git_worktree_clean": status == "",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "original_scripts": [
            {"path": str(p), "sha256": sha256(p),
             "git_tracked_revision_blob": git(REPO, "rev-parse",
                                              f"HEAD:{pathlib.Path(p).relative_to(REPO).as_posix()}")
             if str(p).startswith(str(REPO)) else None}
            for p in original_scripts],
        "commands": commands,
        "environment": environment(),
    }
    if scratch_copies:
        d["scratch_copies"] = [{"path": str(p), "sha256": sha256(p)}
                               for p in scratch_copies]
        d["scratch_copy_modification"] = (
            'Only change: sys.path.insert(0, r"C:\\Users\\amirh\\Desktop\\'
            'qwarm-gnn-rl\\src") -> sys.path.insert(0, r"C:\\Users\\amirh\\'
            'Desktop\\Thesis_Project\\src")')
    if extra:
        d.update(extra)
    return d


def run_copy(name, cwd=None):
    """Run a scratch copy, return (command string, stdout)."""
    script = SCRATCH / name
    cmd = [PY, str(script)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or SCRATCH)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return " ".join(f'"{c}"' if " " in c else c for c in cmd), r.stdout


def write(name, obj):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print("wrote", p)
    return p
