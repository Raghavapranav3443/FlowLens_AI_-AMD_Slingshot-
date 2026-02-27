"""
FlowLens AI — One-Click Launcher
Run from project root: python start.py
Logs from backend and frontend stream directly into this terminal.
Press Ctrl+C to stop everything.
"""

import subprocess
import sys
import os
import time
import threading
import urllib.request
import webbrowser

# Force UTF-8 output on Windows so emoji from subprocesses don't crash
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

# ── CONFIG ───────────────────────────────────────────────────────────────────
BACKEND_DIR   = "backend"
FRONTEND_DIR  = "frontend"
OLLAMA_MODEL  = "llama3.2:3b"
BACKEND_PORT  = 8000
FRONTEND_PORT = 3000
# ─────────────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))

# Load .env from project root
def load_env():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        print("  !!  No .env file found in project root.")
        print("      Create one with:  GEMINI_API_KEY=your_key_here")
        return {}
    vals = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                vals[key.strip()] = val.strip().strip('"').strip("'")
    return vals

env_vars = load_env()
GEMINI_API_KEY = env_vars.get("GEMINI_API_KEY", "")
processes = []

def banner(msg):
    print(f"\n  {'='*44}")
    print(f"   {msg}")
    print(f"  {'='*44}\n")

def ok(msg):   print(f"  OK  {msg}")
def warn(msg): print(f"  !!  {msg}")
def info(msg): print(f"      {msg}")

def fail(msg):
    print(f"\n  XX  {msg}\n")
    sys.exit(1)

def check_command(cmd):
    try:
        subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def find_npm():
    if check_command("npm"):
        return "npm"
    for p in [
        os.path.expandvars(r"%ProgramFiles%\nodejs\npm.cmd"),
        os.path.expandvars(r"%ProgramFiles(x86)%\nodejs\npm.cmd"),
        os.path.expandvars(r"%APPDATA%\npm\npm.cmd"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\nodejs\npm.cmd"),
    ]:
        if os.path.exists(p):
            return p
    return None

def find_uvicorn(backend_path):
    venv_uv = os.path.join(backend_path, "venv", "Scripts", "uvicorn.exe")
    if os.path.exists(venv_uv):
        return venv_uv
    if check_command("uvicorn"):
        return "uvicorn"
    return None

def backend_healthy():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout=2)
        return True
    except:
        return False

def stream_output(proc, prefix):
    """Stream process output into this terminal with a label prefix."""
    try:
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[{prefix}] {text}")
    except:
        pass

def kill_all():
    for p in processes:
        try:
            p.terminate()
        except:
            pass

def ensure_npm_packages(frontend_path, npm_cmd):
    """Run npm install if node_modules is missing or package.json is newer than node_modules."""
    node_modules = os.path.join(frontend_path, "node_modules")
    package_json = os.path.join(frontend_path, "package.json")

    needs_install = False

    if not os.path.exists(node_modules):
        info("node_modules not found -- running npm install...")
        needs_install = True
    elif os.path.getmtime(package_json) > os.path.getmtime(node_modules):
        info("package.json has changed -- running npm install to sync...")
        needs_install = True

    if needs_install:
        result = subprocess.run(
            [npm_cmd, "install", "--no-audit", "--no-fund"],
            cwd=frontend_path,
            shell=(sys.platform == "win32"),
        )
        if result.returncode != 0:
            fail("npm install failed. Check the output above for errors.")
        ok("npm install complete.")
    else:
        ok("node_modules up to date -- skipping npm install.")

# ── MAIN ─────────────────────────────────────────────────────────────────────
banner("FlowLens AI -- One-Click Launcher")

# 1. CHECKS
print("[1/6] Checking prerequisites...")

if not check_command("ollama"):
    fail("Ollama not found. Install from https://ollama.ai/download")
ok("Ollama found.")

npm_cmd = find_npm()
if not npm_cmd:
    fail("npm not found. Install Node.js from https://nodejs.org")
ok("npm found.")

backend_path  = os.path.join(ROOT, BACKEND_DIR)
frontend_path = os.path.join(ROOT, FRONTEND_DIR)

if not os.path.exists(backend_path):
    fail(f"Backend folder not found: {backend_path}  -- check BACKEND_DIR in start.py")
if not os.path.exists(os.path.join(frontend_path, "package.json")):
    fail(f"Frontend package.json not found in: {frontend_path}  -- check FRONTEND_DIR in start.py")

uvicorn_cmd = find_uvicorn(backend_path)
if not uvicorn_cmd:
    fail("uvicorn not found. Activate your venv and run: pip install uvicorn")
ok("uvicorn found.")

if not GEMINI_API_KEY:
    warn("GEMINI_API_KEY missing from .env -- cloud mode won't work. AMD local mode is unaffected.")
    time.sleep(1)

# 2. NPM INSTALL
print("\n[2/6] Checking frontend dependencies...")
ensure_npm_packages(frontend_path, npm_cmd)

# 3. OLLAMA
print("\n[3/6] Starting Ollama...")

ollama_check = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
if ollama_check.returncode != 0:
    ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    processes.append(ollama_proc)
    threading.Thread(target=stream_output, args=(ollama_proc, "ollama"), daemon=True).start()
    time.sleep(3)
    ok("Ollama server started.")
else:
    ok("Ollama already running.")

info(f"Ensuring {OLLAMA_MODEL} is available...")
model_list = subprocess.run(["ollama", "list"], capture_output=True, text=True)
if OLLAMA_MODEL.split(":")[0] not in model_list.stdout:
    info(f"Pulling {OLLAMA_MODEL} -- this may take a few minutes...")
    subprocess.run(["ollama", "pull", OLLAMA_MODEL])
ok("Model ready.")

# 4. BACKEND
print("\n[4/6] Starting backend...")

env_backend = os.environ.copy()
env_backend["GEMINI_API_KEY"] = GEMINI_API_KEY

backend_proc = subprocess.Popen(
    [uvicorn_cmd, "main:app", "--reload", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
    cwd=backend_path,
    env=env_backend,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
processes.append(backend_proc)
threading.Thread(target=stream_output, args=(backend_proc, "backend"), daemon=True).start()
ok("Backend process launched.")

info("Waiting for backend to come online...")
for _ in range(40):
    if backend_healthy():
        break
    if backend_proc.poll() is not None:
        fail("Backend crashed on startup. Check the [backend] log above for errors.")
    time.sleep(2)
else:
    fail("Backend didn't respond within 80s. Check the [backend] log above.")
ok(f"Backend is online  -->  http://127.0.0.1:{BACKEND_PORT}")

# 5. FRONTEND
print("\n[5/6] Starting frontend...")

env_frontend = os.environ.copy()
env_frontend["BROWSER"] = "none"  # we open the browser ourselves below

frontend_proc = subprocess.Popen(
    [npm_cmd, "start"],
    cwd=frontend_path,
    env=env_frontend,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    shell=(sys.platform == "win32"),
)
processes.append(frontend_proc)
threading.Thread(target=stream_output, args=(frontend_proc, "frontend"), daemon=True).start()
ok("Frontend process launched.")

info("Waiting for React to compile (this takes ~10s the first time)...")
time.sleep(10)

# 6. OPEN BROWSER + KEEP ALIVE
print(f"\n[6/6] All services running!\n")
print(f"  Backend   -->  http://127.0.0.1:{BACKEND_PORT}")
print(f"  Frontend  -->  http://localhost:{FRONTEND_PORT}")
print(f"  API Docs  -->  http://127.0.0.1:{BACKEND_PORT}/docs")
print(f"\n  Logs streaming below. Press Ctrl+C to stop everything.\n")
print(f"  {'-'*44}")

webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

try:
    while True:
        if backend_proc.poll() is not None:
            warn("Backend stopped unexpectedly -- check [backend] logs above.")
        time.sleep(5)
except KeyboardInterrupt:
    print("\n\n  Shutting down all services...\n")
    kill_all()
    print("  Done. Goodbye!\n")
    sys.exit(0)