from __future__ import annotations

import atexit
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

# ------------------------------
# Public env toggles (optional)
# ------------------------------
ENV_LOG_LEVEL = "DAIMA_LOG_LEVEL"  # e.g. DEBUG/INFO/WARNING
ENV_DEBUG_OUT_DIR = "DAIMA_DEBUG_OUT_DIR"  # absolute path or path relative to repo root
ENV_DISABLE_FILE_LOG = "DAIMA_DISABLE_FILE_LOG"  # 1/true => no log file
ENV_DISABLE_TEE = "DAIMA_DISABLE_TEE"  # 1/true => don't wrap stdout/stderr

# Cleanup policy:
# - Default: keep all logs (append-only).
# - Optional: allow enabling cleanup via env if a user explicitly wants it.
ENV_ENABLE_CLEANUP = "DAIMA_ENABLE_CLEANUP"  # 1/true => enable cleanup
ENV_KEEP_DAYS = "DAIMA_KEEP_DAYS"  # override keep days (int), default ~2 months
DEFAULT_KEEP_DAYS = 62


def _is_truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_mkdir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Never block the main program on debug artifact mkdir failures.
        pass


def _find_repo_root(from_path: Path) -> Path:
    """
    Best-effort repo root detection.

    We anchor debug_out under the repo root (where AGENTS.md lives) so logs are
    stable regardless of the current working directory / PyCharm Run config.
    """
    from_path = from_path.resolve()
    candidates = [from_path] + list(from_path.parents)
    for d in candidates:
        if (d / ".release-root").is_file():
            return d
    for d in candidates:
        if (d / ".git").exists():
            return d
    return from_path.parent


def _now_run_id() -> str:
    # Include microseconds + pid to avoid collisions when running multiple times quickly.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{ts}_{os.getpid()}"


class _TeeTextStream:
    def __init__(self, original, log_fp, lock: threading.Lock, name: str) -> None:
        self._original = original
        self._log_fp = log_fp
        self._lock = lock
        self._name = name

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8") or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._original, "errors", "backslashreplace") or "backslashreplace"

    def isatty(self) -> bool:
        try:
            return bool(self._original.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def buffer(self):
        # Some libraries write to sys.stdout.buffer (bytes). We proxy it, but we also
        # mirror bytes into the text log file (best-effort decode).
        buf = getattr(self._original, "buffer", None)
        if buf is None:
            raise AttributeError("buffer")
        return _TeeBinaryStream(buf, self._log_fp, self._lock, self._name)

    def write(self, s) -> int:
        if s is None:
            return 0
        if isinstance(s, bytes):
            try:
                s = s.decode(self.encoding, self.errors)
            except Exception:
                s = repr(s)
        with self._lock:
            n = 0
            try:
                n = int(self._original.write(s) or 0)
            except Exception:
                # Ignore console write failures; keep trying to write the log file.
                n = 0
            try:
                self._log_fp.write(s)
            except Exception:
                pass
        return n

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        with self._lock:
            try:
                self._original.flush()
            except Exception:
                pass
            try:
                self._log_fp.flush()
            except Exception:
                pass


class _TeeBinaryStream:
    def __init__(self, original_buffer, log_fp, lock: threading.Lock, name: str) -> None:
        self._original = original_buffer
        self._log_fp = log_fp
        self._lock = lock
        self._name = name

    def write(self, b: bytes) -> int:
        if b is None:
            return 0
        if not isinstance(b, (bytes, bytearray)):
            b = str(b).encode("utf-8", "backslashreplace")
        with self._lock:
            n = 0
            try:
                n = int(self._original.write(b) or 0)
            except Exception:
                n = 0
            try:
                self._log_fp.write(b.decode("utf-8", "backslashreplace"))
            except Exception:
                pass
        return n

    def flush(self) -> None:
        with self._lock:
            try:
                self._original.flush()
            except Exception:
                pass
            try:
                self._log_fp.flush()
            except Exception:
                pass


_SENSITIVE_KEY_RE = re.compile(r"(pass(word)?|token|secret|api[_-]?key|auth)", re.IGNORECASE)


def _scrub(obj: Any) -> Any:
    """Make objects JSON-safe and scrub obvious secrets."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if _SENSITIVE_KEY_RE.search(ks):
                out[ks] = "***"
            else:
                out[ks] = _scrub(v)
        return out
    # Best-effort stringification for small / simple objects.
    try:
        s = str(obj)
        if len(s) <= 500:
            return s
    except Exception:
        pass
    return {"__repr__": repr(obj)[:500]}


def _write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: On this repo's typical Windows setup, delete/rename can be denied by ACLs.
        # So we avoid atomic "write tmp then rename" here and write directly.
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort; never block main program.
        pass


def _cleanup_old_artifacts(debug_out_dir: Path, keep_days: int = DEFAULT_KEEP_DAYS) -> None:
    """Delete log/run artifacts older than keep_days (mtime-based) under debug_out."""
    if keep_days <= 0:
        return
    try:
        debug_out_dir = debug_out_dir.resolve()
    except Exception:
        return

    cutoff = time.time() - keep_days * 86400
    for sub in ("logs", "runs"):
        base = debug_out_dir / sub
        if not base.exists():
            continue
        # Delete old files.
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    # Avoid touching non-text/binary artifacts that might live under debug_out.
                    if p.suffix.lower() not in (".log", ".json", ".txt"):
                        continue
                    if p.stat().st_mtime < cutoff:
                        # Best-effort only. If ACLs forbid delete, we keep the file unchanged.
                        p.unlink(missing_ok=True)
            except Exception:
                pass
        # Directory removal is best-effort and may be denied by ACLs; skipping is harmless.


def _best_effort_git_info(repo_root: Path) -> Dict[str, Any]:
    # Avoid hard dependency on git; time-bounded.
    import subprocess

    out: Dict[str, Any] = {}
    if not (repo_root / ".git").exists():
        return out
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if r.returncode == 0:
            out["git_head"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if r.returncode == 0:
            out["git_dirty"] = bool(r.stdout.strip())
    except Exception:
        pass
    return out


@dataclass
class RunContext:
    repo_root: Path
    debug_out_dir: Path
    run_id: str
    log_path: Optional[Path]
    run_dir: Optional[Path]
    _paths: Dict[str, Any] = field(default_factory=dict)
    _config: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot_paths(
        self,
        *,
        inputs: Optional[Mapping[str, Any]] = None,
        outputs: Optional[Mapping[str, Any]] = None,
        config_files: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.run_dir is None:
            return
        payload: Dict[str, Any] = dict(self._paths)
        if inputs:
            payload.setdefault("inputs", {}).update(_scrub(dict(inputs)))
        if outputs:
            payload.setdefault("outputs", {}).update(_scrub(dict(outputs)))
        if config_files:
            payload.setdefault("config_files", {}).update(_scrub(dict(config_files)))
        if extra:
            payload.setdefault("extra", {}).update(_scrub(dict(extra)))
        self._paths = payload
        _write_json(self.run_dir / "paths.json", payload)

    def snapshot_config(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.run_dir is None:
            return
        payload: Dict[str, Any] = dict(self._config)
        if config:
            payload.setdefault("config", {}).update(_scrub(dict(config)))
        if extra:
            payload.setdefault("extra", {}).update(_scrub(dict(extra)))
        self._config = payload
        _write_json(self.run_dir / "config_snapshot.json", payload)


_GLOBAL_CTX: Optional[RunContext] = None


def setup_run(script_path: str) -> RunContext:
    """
    Initialize unified console logging + debug snapshots for a script.

    - Mirrors stdout/stderr into a log file under debug_out/logs/...
    - Writes run metadata + path/config snapshots under debug_out/runs/<run_id>/...
    - Deletes artifacts older than ~2 months under debug_out/{logs,runs}/ (mtime-based).

    This module is stdlib-only and should be safe across different virtualenv/conda envs.

    References:
      - sys.excepthook: https://docs.python.org/3/library/sys.html#sys.excepthook
      - logging.basicConfig(force=...): https://docs.python.org/3/library/logging.html#logging.basicConfig
      - atexit.register: https://docs.python.org/3/library/atexit.html
      - faulthandler.enable(file=...): https://docs.python.org/3/library/faulthandler.html
    """
    global _GLOBAL_CTX
    if _GLOBAL_CTX is not None:
        return _GLOBAL_CTX

    script_p = Path(script_path).resolve()
    repo_root = _find_repo_root(script_p)

    # debug_out location: env override (relative to repo root) else <repo_root>/debug_out.
    debug_out_raw = os.environ.get(ENV_DEBUG_OUT_DIR)
    if debug_out_raw:
        debug_out_dir = Path(debug_out_raw)
        if not debug_out_dir.is_absolute():
            debug_out_dir = repo_root / debug_out_dir
    else:
        debug_out_dir = repo_root / "debug_out"

    _safe_mkdir(debug_out_dir)

    # Optional cleanup (disabled by default; logs are append-only unless user enables cleanup).
    if _is_truthy_env(ENV_ENABLE_CLEANUP):
        keep_days = DEFAULT_KEEP_DAYS
        raw_keep = os.environ.get(ENV_KEEP_DAYS)
        if raw_keep:
            try:
                keep_days = int(raw_keep)
            except Exception:
                keep_days = DEFAULT_KEEP_DAYS
        _cleanup_old_artifacts(debug_out_dir, keep_days=keep_days)

    run_id = _now_run_id()

    # Build per-script log dir: debug_out/logs/<relative_dir>/<script_stem>/
    try:
        rel = script_p.relative_to(repo_root)
        rel_dir = Path(*rel.parts[:-1])
        stem = rel.stem
    except Exception:
        rel_dir = Path()
        stem = script_p.stem

    log_path: Optional[Path] = None
    run_dir: Optional[Path] = None
    log_fp = None

    if not _is_truthy_env(ENV_DISABLE_FILE_LOG):
        log_dir = debug_out_dir / "logs" / rel_dir / stem
        run_dir = debug_out_dir / "runs" / run_id
        _safe_mkdir(log_dir)
        _safe_mkdir(run_dir)
        log_path = log_dir / f"{run_id}.log"
        try:
            log_fp = open(log_path, "a", encoding="utf-8", errors="backslashreplace")
        except Exception:
            log_fp = None

    ctx = RunContext(
        repo_root=repo_root,
        debug_out_dir=debug_out_dir,
        run_id=run_id,
        log_path=log_path,
        run_dir=run_dir,
    )
    # Ensure snapshot files exist even if a script forgets to call snapshot_* explicitly.
    if run_dir is not None:
        ctx._paths = {"inputs": {}, "outputs": {}, "config_files": {}, "extra": {}}
        ctx._config = {"config": {}, "extra": {}}
    _GLOBAL_CTX = ctx

    # Prepare a lock shared between stdout/stderr tee and extra writers.
    lock = ctx._lock

    # Header banner (best-effort).
    if log_fp is not None:
        try:
            header = [
                "",
                "=" * 88,
                f"RUN START {datetime.now().isoformat(timespec='seconds')}  run_id={run_id}",
                f"script={script_p}",
                f"cwd={Path.cwd().resolve()}",
                f"python={sys.version.replace(os.linesep, ' ')}",
                f"executable={sys.executable}",
                "=" * 88,
                "",
            ]
            with lock:
                log_fp.write("\n".join(header))
                log_fp.flush()
        except Exception:
            pass

    # Tee stdout/stderr to file (user can disable via env).
    if (log_fp is not None) and (not _is_truthy_env(ENV_DISABLE_TEE)):
        try:
            sys.stdout = _TeeTextStream(sys.stdout, log_fp, lock, name="stdout")  # type: ignore[assignment]
        except Exception:
            pass
        try:
            sys.stderr = _TeeTextStream(sys.stderr, log_fp, lock, name="stderr")  # type: ignore[assignment]
        except Exception:
            pass

    # Configure logging (only if the app hasn't configured handlers yet).
    level_name = os.environ.get(ENV_LOG_LEVEL, "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Hook sys.excepthook to add a banner + flush (avoid double-print by delegating to original).
    orig_excepthook = sys.excepthook

    def _hooked_excepthook(exctype, value, tb):  # type: ignore[no-untyped-def]
        try:
            print("\n[DAIMA] Uncaught exception (sys.excepthook):", file=sys.stderr)
        except Exception:
            pass
        try:
            orig_excepthook(exctype, value, tb)
        finally:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            try:
                sys.stdout.flush()
            except Exception:
                pass

    sys.excepthook = _hooked_excepthook  # type: ignore[assignment]

    # Hook threading.excepthook (Python 3.8+) for background thread errors.
    try:
        import threading as _threading

        orig_thread_hook = getattr(_threading, "excepthook", None)

        if callable(orig_thread_hook):

            def _hooked_thread_excepthook(args):  # type: ignore[no-untyped-def]
                try:
                    print(
                        f"\n[DAIMA] Uncaught thread exception: {getattr(args, 'thread', None)}",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
                try:
                    orig_thread_hook(args)
                finally:
                    try:
                        sys.stderr.flush()
                    except Exception:
                        pass

            _threading.excepthook = _hooked_thread_excepthook  # type: ignore[assignment]
    except Exception:
        pass

    # Enable faulthandler to dump tracebacks on crashes / deadlocks (best-effort).
    if log_fp is not None:
        try:
            import faulthandler

            # Docs: the file must be kept open for faulthandler.enable(file=...).
            faulthandler.enable(file=log_fp, all_threads=True)
        except Exception:
            pass

    # Run metadata (write early).
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_path": str(script_p),
        "cwd": str(Path.cwd().resolve()),
        "argv": list(sys.argv),
        "python_version": sys.version,
        "executable": sys.executable,
        "sys_prefix": getattr(sys, "prefix", None),
        "sys_base_prefix": getattr(sys, "base_prefix", None),
        "sys_real_prefix": getattr(sys, "real_prefix", None),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "pythonpath_env": os.environ.get("PYTHONPATH"),
    }
    meta.update(_best_effort_git_info(repo_root))

    if run_dir is not None:
        _write_json(run_dir / "run_meta.json", _scrub(meta))
        _write_json(run_dir / "paths.json", _scrub(ctx._paths))
        _write_json(run_dir / "config_snapshot.json", _scrub(ctx._config))

    # Best-effort automatic snapshot at exit (pull simple-looking globals from __main__).
    def _auto_snapshot_at_exit() -> None:
        if ctx.run_dir is None:
            return
        try:
            import __main__

            g = vars(__main__)
        except Exception:
            g = {}

        # Paths: common variable naming patterns.
        path_keys = []
        for k, v in g.items():
            if not isinstance(k, str):
                continue
            ku = k.upper()
            if any(x in ku for x in ("PATH", "DIR", "FILE", "INPUT", "OUTPUT")):
                path_keys.append(k)

        paths_payload: Dict[str, Any] = dict(ctx._paths)
        auto_paths: Dict[str, Any] = {}
        for k in sorted(set(path_keys)):
            v = g.get(k)
            if isinstance(v, (str, Path)):
                auto_paths[k] = str(v)
        if auto_paths:
            paths_payload.setdefault("auto_globals", {}).update(_scrub(auto_paths))
            ctx._paths = paths_payload
            _write_json(ctx.run_dir / "paths.json", paths_payload)

        # Config: capture small dict-like globals.
        cfg_payload: Dict[str, Any] = dict(ctx._config)
        auto_cfg: Dict[str, Any] = {}
        for k, v in g.items():
            if not isinstance(k, str):
                continue
            ku = k.upper()
            if any(x in ku for x in ("CONFIG", "CFG", "SETTINGS", "PARAM")):
                if isinstance(v, dict):
                    # Avoid writing huge configs.
                    if len(v) <= 200:
                        auto_cfg[k] = v
        if auto_cfg:
            cfg_payload.setdefault("auto_globals", {}).update(_scrub(auto_cfg))
            ctx._config = cfg_payload
            _write_json(ctx.run_dir / "config_snapshot.json", cfg_payload)

        # Update last_run pointer.
        last = {
            "run_id": ctx.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "log_path": str(ctx.log_path) if ctx.log_path else None,
            "run_dir": str(ctx.run_dir) if ctx.run_dir else None,
        }
        _write_json(ctx.debug_out_dir / "last_run.json", _scrub(last))

        # Final flush.
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            if log_fp is not None:
                with lock:
                    log_fp.flush()
        except Exception:
            pass

    atexit.register(_auto_snapshot_at_exit)

    return ctx
