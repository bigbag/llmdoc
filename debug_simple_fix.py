import asyncio
import json
import logging
import subprocess
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_debug_session():
    # Path to python executable
    python_exe = sys.executable
    if ".venv" not in python_exe:
        # Try to find venv python
        venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
        if os.path.exists(venv_python):
            python_exe = venv_python

    print(f"Using python: {python_exe}")

    cmd = [python_exe, "-m", "llmdoc"]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    print(f"Async Server started with PID: {proc.pid}")

    async def log_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            print(f"[STDERR] {line.decode().strip()}")

    asyncio.create_task(log_stderr())

    async def send_json(data):
        msg = json.dumps(data) + "\n"
        proc.stdin.write(msg.encode())
        await proc.stdin.drain()
        print(f"Sent: {json.dumps(data)}")

    # 1. Initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "debugger", "version": "0.1.0"},
        },
    }
    await send_json(init_req)

    # Read response
    # Increase limit for large responses
    proc.stdout._limit = 1024 * 1024 * 10  # 10MB limit
    line = await proc.stdout.readline()
    print(f"Received: {line.decode().strip()}")

    # 2. Initialized
    await send_json({"jsonrpc": "2.0", "method": "notifications/initialized"})

    # 3. Call get_doc (stress test)
    get_doc_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_doc",
            "arguments": {"url": "https://ai.pydantic.dev/agents/index.md"},
        },
    }
    await send_json(get_doc_req)

    print("Waiting for get_doc response...")
    line = await proc.stdout.readline()
    if line:
        print(f"Get Doc Result: {line.decode().strip()[:200]}...")
    else:
        print("No response for get_doc (Connection closed?)")

    proc.terminate()


if __name__ == "__main__":
    asyncio.run(run_debug_session())
