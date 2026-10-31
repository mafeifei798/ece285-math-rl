"""Bootstrap an isolated Python 3.11 environment without touching other projects."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
venv = root / ".venv"
receipt = root / "receipts/environment_install.json"
env = os.environ.copy()
proxy = os.environ.get("ECE285_HTTP_PROXY", "http://127.0.0.1:7890")
for key in list(env):
    if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "pip_proxy"}:
        env.pop(key)
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    env[key] = proxy
env.update(NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1")
env.update(PIP_CACHE_DIR=str(root / "cache/pip"), PIP_DISABLE_PIP_VERSION_CHECK="1")
env["PIP_INDEX_URL"] = os.environ.get("ECE285_PIP_INDEX_URL", "https://pypi.org/simple")
env["PIP_CONFIG_FILE"] = os.devnull
env.update(PIP_DEFAULT_TIMEOUT="60", PIP_RETRIES="8", PIP_RESUME_RETRIES="10")
state = {"pid": os.getpid(), "started_unix": time.time(), "status": "running", "training_started": False}

def save():
    receipt.write_text(json.dumps(state, indent=2) + "\n")

save()
try:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    # System pip can target a venv without installing into the system interpreter.
    installer = [sys.executable, "-m", "pip", "--python", str(venv / "bin/python")]
    subprocess.run(installer + ["install", "--index-url", env["PIP_INDEX_URL"], "pip==25.3"], check=True, env=env)
    wheels = root / "cache/wheels"
    wheels.mkdir(exist_ok=True)
    resume_plan = root / "receipts/resume_wheels.json"
    if resume_plan.exists():
        for item in json.loads(resume_plan.read_text()):
            target = wheels / item["filename"]
            partial = wheels / (item["filename"] + ".partial")
            if not target.exists():
                subprocess.run(["curl", "--fail", "--location", "--retry", "8", "--retry-all-errors",
                    "--connect-timeout", "20", "--max-time", "1800", "--continue-at", "-", "--output", str(partial), item["url"]],
                    check=True, env=env)
                partial.replace(target)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError("Resumed wheel checksum mismatch: " + item["filename"])
    subprocess.run([str(venv / "bin/python"), "-m", "pip", "install", "--index-url", env["PIP_INDEX_URL"],
                    "--find-links", str(wheels), "--resume-retries", "10", "-r", str(root / "training/requirements.txt")], check=True, env=env)
    freeze = subprocess.check_output([str(venv / "bin/python"), "-m", "pip", "freeze"], env=env, text=True)
    (root / "receipts/requirements.lock.txt").write_text(freeze)
    subprocess.run([str(venv / "bin/python"), "-m", "pip", "check"], check=True, env=env)
    state.update(status="completed", finished_unix=time.time())
except Exception as exc:
    state.update(status="failed", error=str(exc), finished_unix=time.time())
    raise
finally:
    save()
