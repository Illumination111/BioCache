import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
import yaml


def start_run(output, config):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    (path / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        revision = "uncommitted"
    write_json(path / "run.json", {"command": sys.argv, "python": sys.version,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "platform": platform.platform(), "git_revision": revision,
               "working_tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                    capture_output=True, text=True).stdout),
               "cuda_available": torch.cuda.is_available()})
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
