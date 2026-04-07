from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def probe_current_module(module_name: str, *, import_target: str | None = None) -> dict[str, Any]:
    if import_target is None:
        import_target = module_name

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {
            "ok": False,
            "module_name": module_name,
            "import_target": import_target,
            "module_found": False,
            "importable": False,
            "version": None,
            "error": f"Module not found: {module_name}",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        }

    try:
        module = importlib.import_module(import_target)
    except Exception as exc:
        return {
            "ok": False,
            "module_name": module_name,
            "import_target": import_target,
            "module_found": True,
            "importable": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        }

    return {
        "ok": True,
        "module_name": module_name,
        "import_target": import_target,
        "module_found": True,
        "importable": True,
        "version": getattr(module, "__version__", None),
        "error": None,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }


def probe_python_info(
    python_executable: str | Path,
    *,
    timeout_ms: int = 20000,
) -> dict[str, Any]:
    python_path = Path(python_executable)
    if not python_path.exists():
        return {
            "ok": False,
            "python_executable": str(python_path),
            "python_version": None,
            "error": f"Interpreter not found: {python_path}",
        }

    script = "\n".join(
        [
            "import json",
            "import sys",
            "print(json.dumps({'ok': True, 'python_executable': sys.executable, 'python_version': sys.version.split()[0], 'error': None}, ensure_ascii=False))",
        ]
    )

    completed = subprocess.run(
        [str(python_path), "-c", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_ms / 1000)),
    )

    stdout = (completed.stdout or "").strip()
    if not stdout:
        stderr = (completed.stderr or "").strip()
        return {
            "ok": False,
            "python_executable": str(python_path),
            "python_version": None,
            "error": stderr or f"Probe process returned exit code {completed.returncode}",
        }

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        stderr = (completed.stderr or "").strip()
        return {
            "ok": False,
            "python_executable": str(python_path),
            "python_version": None,
            "error": stderr or f"Invalid probe output: {stdout}",
        }

    if completed.returncode not in {0} and payload.get("ok", False):
        payload["ok"] = False
        payload["error"] = payload.get("error") or f"Probe process returned exit code {completed.returncode}"
    return payload


def probe_python_module(
    python_executable: str | Path,
    module_name: str,
    *,
    import_target: str | None = None,
    timeout_ms: int = 20000,
) -> dict[str, Any]:
    python_path = Path(python_executable)
    if not python_path.exists():
        return {
            "ok": False,
            "module_name": module_name,
            "import_target": import_target or module_name,
            "module_found": False,
            "importable": False,
            "version": None,
            "error": f"Interpreter not found: {python_path}",
            "python_executable": str(python_path),
            "python_version": None,
        }

    script = "\n".join(
        [
            "import importlib",
            "import importlib.util",
            "import json",
            "import sys",
            "module_name = sys.argv[1]",
            "import_target = sys.argv[2]",
            "spec = importlib.util.find_spec(module_name)",
            "payload = {'module_name': module_name, 'import_target': import_target, 'python_executable': sys.executable, 'python_version': sys.version.split()[0]}",
            "if spec is None:",
            "    payload.update({'ok': False, 'module_found': False, 'importable': False, 'version': None, 'error': f'Module not found: {module_name}'})",
            "    print(json.dumps(payload, ensure_ascii=False))",
            "    raise SystemExit(0)",
            "try:",
            "    module = importlib.import_module(import_target)",
            "    payload.update({'ok': True, 'module_found': True, 'importable': True, 'version': getattr(module, '__version__', None), 'error': None})",
            "except Exception as exc:",
            "    payload.update({'ok': False, 'module_found': True, 'importable': False, 'version': None, 'error': f'{type(exc).__name__}: {exc}'})",
            "print(json.dumps(payload, ensure_ascii=False))",
        ]
    )

    completed = subprocess.run(
        [str(python_path), "-c", script, module_name, import_target or module_name],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_ms / 1000)),
    )

    stdout = (completed.stdout or "").strip()
    if not stdout:
        stderr = (completed.stderr or "").strip()
        return {
            "ok": False,
            "module_name": module_name,
            "import_target": import_target or module_name,
            "module_found": False,
            "importable": False,
            "version": None,
            "error": stderr or f"Probe process returned exit code {completed.returncode}",
            "python_executable": str(python_path),
            "python_version": None,
        }

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        stderr = (completed.stderr or "").strip()
        return {
            "ok": False,
            "module_name": module_name,
            "import_target": import_target or module_name,
            "module_found": False,
            "importable": False,
            "version": None,
            "error": stderr or f"Invalid probe output: {stdout}",
            "python_executable": str(python_path),
            "python_version": None,
        }

    if completed.returncode not in {0} and payload.get("ok", False):
        payload["ok"] = False
        payload["error"] = payload.get("error") or f"Probe process returned exit code {completed.returncode}"
    return payload


def summarize_probe(prefix: str, probe: dict[str, Any]) -> tuple[bool, str]:
    if probe.get("ok"):
        version = probe.get("version")
        version_suffix = f" (version {version})" if version else ""
        return True, f"{prefix} is importable via {probe.get('python_executable')}{version_suffix}."

    if not probe.get("module_found"):
        return False, f"{prefix} is unavailable: {probe.get('error')}"

    return False, f"{prefix} failed to import via {probe.get('python_executable')}: {probe.get('error')}"
