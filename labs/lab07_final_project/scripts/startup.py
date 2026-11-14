"""Foreground supervisor. Startup never installs packages or downloads models."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml")]


class ChildJob:
    """Windows closes this handle on supervisor death, terminating its child tree."""
    def __init__(self):
        self.handle = None
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
            self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.CreateJobObjectW(None, None)
            # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, x64 Windows Python.
            import struct
            if ctypes.sizeof(ctypes.c_void_p) != 8:
                raise RuntimeError('64-bit Python is required on Windows.')
            limits = ctypes.create_string_buffer(144)
            struct.pack_into('I', limits, 16, 0x2000)  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.handle or not self.kernel.SetInformationJobObject(self.handle, 9, limits, len(limits)):
                self.close()
                raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if self.handle and not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            import ctypes
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def ok(message):
    print(f"[OK] {message}", flush=True)


def command(args, timeout=30):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def wait_ready(check, description, timeout=60, process=None):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Streamlit exited ({process.returncode}); inspect outputs/startup.log.")
        try:
            if check():
                return
        except Exception as exc:
            last = exc
        # Poll interval only; success always requires a positive readiness check.
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {description}: {last or 'not ready'}")


def load_environment():
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.partition('=')
            if not sep or not key.strip().isidentifier():
                raise RuntimeError(f"Invalid .env line: {line}. Use KEY=value.")
            os.environ.setdefault(key.strip(), value.strip().strip('\"\''))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('LANGCHAIN_TRACING_V2', 'false')
    os.environ.setdefault('LANGSMITH_TRACING', 'false')


def dependencies():
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError('Python 3.12-3.13 is required. Run setup.ps1 with a compatible interpreter.')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use a project virtual environment; run setup.ps1 first.')
    ok(f"Python environment: {sys.executable}")
    try:
        from packaging.requirements import Requirement
    except ImportError as exc:
        raise RuntimeError('Dependency packaging is missing. Run .\\setup.ps1.') from exc
    problems = []
    for line in (ROOT / 'requirements.lock.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            version = metadata.version(req.name)
            if version not in req.specifier:
                problems.append(f'{req.name}: installed {version}, required {req.specifier}')
        except metadata.PackageNotFoundError:
            problems.append(f'{req.name}: missing')
    if problems:
        raise RuntimeError('Dependencies invalid: ' + '; '.join(problems) + '. Run .\\setup.ps1.')
    try:
        for name in ('knowledge_copilot', 'streamlit', 'agents', 'redis', 'fastapi', 'multipart', 'langchain_redis'):
            importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(f'{exc}. Run .\\setup.ps1 to install the editable package and dependencies.') from exc
    ok('Dependencies')


def main():
    if sys.platform == 'win32':
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK, interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    print('LAB 7 - ENGINEERING KNOWLEDGE COPILOT', flush=True)
    load_environment()
    os.chdir(ROOT)
    outputs = ROOT / 'outputs'
    outputs.mkdir(exist_ok=True)
    lock = (outputs / 'startup.lock').open('a+b')
    try:
        if sys.platform == 'win32':
            import msvcrt
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f'Lab 7 launcher is already running. Use http://localhost:{args.port} or stop it with Ctrl+C.') from exc
        dependencies()
        from knowledge_copilot.config import LAB_ROOT
        if LAB_ROOT != ROOT:
            raise RuntimeError(f'Wrong editable installation ({LAB_ROOT}); run .\\setup.ps1 in {ROOT}.')
        # Reserve/check the port before costly probes. Never terminate its owner.
        with socket.socket() as port:
            try:
                port.bind(('127.0.0.1', args.port))
            except OSError as exc:
                raise RuntimeError(f'Port {args.port} is occupied. Stop its application or the previous Lab 7 launcher.') from exc
        if not shutil.which('docker'):
            raise RuntimeError('Docker CLI missing. Install Docker Desktop with Compose and start its Linux engine.')
        try:
            command(['docker', 'info'])
            command(COMPOSE + ['version'])
        except Exception as exc:
            raise RuntimeError(f'Docker Desktop is unavailable. Start Docker Desktop and wait for its engine. {exc}') from exc
        ok('Docker')
        print('[..] Loading application services...', flush=True)
        from knowledge_copilot.api.dependencies import get_redis_adapter
        from knowledge_copilot.infrastructure.llm_client import probe_lm_studio_readiness
        redis = get_redis_adapter()
        redis_owned = False
        process = None
        job = ChildJob()
        container_id = None
        try:
            if not redis.ping():
                if redis.redis_url.rstrip('/') not in {'redis://localhost:6380', 'redis://127.0.0.1:6380'}:
                    raise RuntimeError(f'Redis unavailable at {redis.redis_url}; start the configured service.')
                running = command(COMPOSE + ['ps', '--status', 'running', '-q', 'redis'])
                redis_owned = not bool(running)
                # Never pull an image as part of routine startup.
                try:
                    command(['docker', 'image', 'inspect', 'redis/redis-stack:7.4.0-v8'])
                except Exception as exc:
                    raise RuntimeError('Redis Stack image is unavailable. Run docker pull redis/redis-stack:7.4.0-v8 once, then retry.') from exc
                command(COMPOSE + ['up', '-d', '--pull', 'never', 'redis'], timeout=60)
                container_id = command(COMPOSE + ['ps', '-q', 'redis'])
            wait_ready(lambda: redis.ping() and redis.get_client().execute_command('FT._LIST') is not None, 'Redis Stack')
            ok('Redis Stack')
            print('[..] Checking LM Studio inference and tool calling...', flush=True)
            probe = probe_lm_studio_readiness(os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1'), os.getenv('LM_STUDIO_MODEL') or None, timeout=60)
            if not probe.is_ready:
                raise RuntimeError(f'LM Studio {probe.status.value}: {probe.detail}. Start its local server and load a tool-capable chat model; set LM_STUDIO_MODEL to its exact ID.')
            os.environ['LM_STUDIO_MODEL'] = probe.model_id
            ok('LM Studio API')
            ok(f'LLM model: {probe.model_id}')
            print('[..] Validating local embedding cache and knowledge index...', flush=True)
            from knowledge_copilot.bootstrap import initialize_knowledge
            initialize_knowledge()
            ok('Knowledge index')
            from knowledge_copilot.api.dependencies import get_sqlite_repo
            get_sqlite_repo()
            script = ROOT / 'src/knowledge_copilot/ui/streamlit_app.py'
            with (outputs / 'startup.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(script), '--server.address', '127.0.0.1', '--server.port', str(args.port), '--server.headless', 'true', '--browser.gatherUsageStats', 'false'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job.assign(process)
                def frontend_ready():
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/_stcore/health', timeout=2) as response:
                        return response.status == 200 and response.read().strip() == b'ok'
                wait_ready(frontend_ready, 'Streamlit', process=process)
                ok('Application ready')
                print(f'Web UI: http://localhost:{args.port}\nStop: Ctrl+C in this terminal.', flush=True)
                if not args.no_browser:
                    webbrowser.open(f'http://localhost:{args.port}')
                process.wait()
                if process.returncode:
                    raise RuntimeError('Streamlit stopped unexpectedly; inspect outputs/startup.log.')
        finally:
            if process is not None and process.poll() is None:
                # Streamlit is a single Python server (no reload subprocess). Kill only our child.
                if sys.platform == 'win32':
                    job.close()
                    process.wait(timeout=10)
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
            job.close()
            if redis_owned:
                container_id = container_id or command(COMPOSE + ['ps', '-q', 'redis'])
                if container_id:
                    command(['docker', 'stop', container_id], timeout=30)
    finally:
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Persistent Redis and SQLite data retained.', flush=True)
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
