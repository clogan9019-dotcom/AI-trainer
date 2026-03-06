import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)

repo_url = "https://github.com/clogan9019-dotcom/AI-trainer.git"
base_dir = Path.cwd()
repo_dir = base_dir / "AI-trainer"

if not repo_dir.exists():
    run(["git", "clone", repo_url, str(repo_dir)])
else:
    print(f"Repo already exists at {repo_dir}")

req = repo_dir / "requirements_cuda.txt"
if req.exists():
    run([sys.executable, "-m", "pip", "install", "-r", str(req), "--index-url", "https://download.pytorch.org/whl/cu121"], cwd=str(repo_dir))
else:
    raise FileNotFoundError(f"Missing requirements file: {req}")
