#!/usr/bin/env python3
# pybrain.brain — minimal, debuggable request-fulfillment runner.
#
# Key properties:
# - The model ALWAYS returns a single executable Python script as Output.
# - The runner executes it, captures full stdout/stderr to disk, and shows a bounded head/tail view in History.
# - "Observations (active)" is a working set (goal + opened dirs/files) that persists until closed.
# - Multi-step plans: on success, AIRESULT_STEPS_REMAINING>0 triggers another invocation with Input: "Continue".
#
# Default: interactive child execution ON (stdin inherited; output mirrored to console + log files).

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .fewshot_openai import few_shot_openai_streaming


DEFAULT_GENERAL_PURPOSE = (
    "You are an agent that ALWAYS writes a single executable Python 3 script as the Output. "
    "The script should perform the task, or ask clarifying questions by printing and then exiting. "
    "When you need filesystem context, emit AIRESULT tags that request the runner to add directory listings "
    "or file snapshots into the Observations section (use the same conventions as in the provided examples). "
    "For multi-step work, emit AIRESULT_STEPS_REMAINING>0 to request a follow-up 'Continue' run, and 0 when done."
)


@dataclass
class RunnerConfig:
    workspace_root: Path = field(default_factory=Path.cwd)
    session_root: Path = field(default_factory=lambda: Path.cwd() / ".runner_session")
    python_exe: str = sys.executable

    # History rendering budgets
    log_head_lines: int = 20
    log_tail_lines: int = 20
    max_line_chars: int = 400
    max_stream_chars: int = 8000

    # Script preview budgets
    script_preview_head: int = 3
    script_preview_tail: int = 2
    script_preview_max_chars: int = 2000

    # File snapshot budgets (Observations)
    file_head_lines: int = 60
    file_tail_lines: int = 60
    file_max_line_chars: int = 400

    # Control loop
    max_auto_steps: int = 12
    timeout_seconds: Optional[int] = None

    # Default: allow interactive child programs
    interactive_child: bool = True


@dataclass
class FileSnapshot:
    path: str
    head: List[str]
    tail: List[str]
    truncated_lines: int
    sha256: str


@dataclass
class DirListing:
    path: str
    entries: List[str]


@dataclass
class Observations:
    # Only current active working-set state (not historical).
    active_goal: Optional[str] = None
    open_dirs: Dict[str, DirListing] = field(default_factory=dict)
    open_files: Dict[str, FileSnapshot] = field(default_factory=dict)

    def render(self) -> str:
        out = ["Observations (active):"]
        if self.active_goal is not None:
            out.append("  OPENGOAL ->")
            out.append(f"    {self.active_goal}")

        for d in self.open_dirs.values():
            out.append(f"  OPENDIR {d.path} ->")
            for ent in d.entries:
                out.append(f"    - {ent}")

        for f in self.open_files.values():
            out.append(f"  OPENFILE {f.path} ->")
            out.extend([f"    {ln}" for ln in f.head])
            if f.truncated_lines > 0:
                out.append(f"    [ {f.truncated_lines} lines truncated ]")
            out.extend([f"    {ln}" for ln in f.tail])
            if f.sha256:
                out.append(f"    (sha256: {f.sha256})")

        if len(out) == 1:
            out.append("  (none)")
        return "\n".join(out)


@dataclass
class RunRecord:
    index: int
    input_text: str
    script_text: str
    ok: bool
    exit_code: int
    duration_ms: int
    fields: Dict[str, List[str]]
    stdout_render: str
    stderr_render: str


# ----------------------------
# Rendering helpers (bounded)
# ----------------------------

def _truncate_line(line: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(line) <= max_chars:
        return line
    cut = max_chars
    removed = len(line) - cut
    return line[:cut] + f" … [truncated {removed} chars]"


def _split_head_tail(lines: List[str], head_n: int, tail_n: int) -> Tuple[List[str], List[str], int]:
    if len(lines) <= head_n + tail_n:
        return lines, [], 0
    head = lines[:head_n]
    tail = lines[-tail_n:]
    trunc = len(lines) - (len(head) + len(tail))
    return head, tail, trunc


def _truncate_text_by_chars(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    marker = "\n      [ ... truncated ... ]\n"
    budget = max_chars - len(marker)
    if budget <= 0:
        return s[:max_chars]
    head = budget // 2
    tail = budget - head
    return s[:head] + marker + s[-tail:]


def render_stream(text: str, head_n: int, tail_n: int, max_line_chars: int, max_total_chars: int) -> str:
    if text == "":
        return "      (empty)"

    raw_lines = text.splitlines()
    lines = [_truncate_line(ln, max_line_chars) for ln in raw_lines]

    head, tail, trunc = _split_head_tail(lines, head_n, tail_n)
    out = [f"      {ln}" for ln in head]
    if trunc > 0:
        out.append(f"      [ {trunc} lines truncated ]")
        out.extend([f"      {ln}" for ln in tail])

    rendered = "\n".join(out)
    return _truncate_text_by_chars(rendered, max_total_chars)


def render_script_preview(script: str, head_n: int, tail_n: int, max_total_chars: int) -> str:
    lines = script.splitlines()
    if not lines:
        return "      (empty)"

    if len(lines) <= head_n + tail_n:
        rendered = "\n".join([f"      {ln}" for ln in lines])
        return _truncate_text_by_chars(rendered, max_total_chars)

    head = lines[:head_n]
    tail = lines[-tail_n:] if tail_n > 0 else []
    trunc = len(lines) - (len(head) + len(tail))
    out = [f"      {ln}" for ln in head]
    out.append(f"      [ truncated {trunc} lines ]")
    out.extend([f"      {ln}" for ln in tail])
    rendered = "\n".join(out)
    return _truncate_text_by_chars(rendered, max_total_chars)


def render_history(cfg: RunnerConfig, records: List[RunRecord]) -> str:
    if not records:
        return "History: (none)"
    out = ["History (most recent last):", ""]
    for r in records:
        out.append(f"[{r.index}] Input: {r.input_text!r}")
        out.append("    Output:")
        out.append(render_script_preview(r.script_text, cfg.script_preview_head, cfg.script_preview_tail, cfg.script_preview_max_chars))
        out.append(f"    Run: {'OK' if r.ok else 'FAIL'}  exit={r.exit_code}")
        out.append("    Fields: (none)" if not r.fields else "    Fields:")
        if r.fields:
            for k in sorted(r.fields.keys()):
                vals = r.fields[k]
                out.append(f"      {k}: {vals[-1]!r}" if len(vals) == 1 else f"      {k}: {vals!r}")
        out.append("    stdout:")
        out.append(r.stdout_render)
        out.append("    stderr:")
        out.append(r.stderr_render)
        out.append("")
    return "\n".join(out).rstrip()


# ----------------------------
# AIRESULT extraction + observation side-effects
# ----------------------------

_AIRESULT_TAG_RE = re.compile(r"<(AIRESULT_[A-Z0-9_]+)>(.*?)</\1>", re.DOTALL)



def extract_fields(stdout_text: str, stderr_text: str) -> Dict[str, List[str]]:
    combined = stdout_text + "\n" + stderr_text
    out: Dict[str, List[str]] = {}
    for tag, val in _AIRESULT_TAG_RE.findall(combined):
        name = tag.replace("AIRESULT_", "")
        out.setdefault(name, []).append(val.strip("\n"))
    return out


def _get_int_field(fields: Dict[str, List[str]], name: str, default: int = 0) -> int:
    try:
        return int(fields.get(name, [str(default)])[-1].strip())
    except Exception:
        return default


def _get_bool_field(fields: Dict[str, List[str]], name: str) -> bool:
    v = fields.get(name, ["false"])[-1].strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def apply_observation_side_effects(cfg: RunnerConfig, obs: Observations, fields: Dict[str, List[str]]) -> None:
    # GOAL
    if "OPENGOAL" in fields:
        obs.active_goal = fields["OPENGOAL"][-1]
    if _get_bool_field(fields, "CLOSEGOAL"):
        obs.active_goal = None

    # DIRS
    for d in fields.get("OPENDIR", []):
        p = Path(d)
        try:
            entries = sorted([x.name + ("/" if x.is_dir() else "") for x in p.iterdir()])
        except FileNotFoundError:
            entries = ["(missing)"]
        except Exception as e:
            entries = [f"(error: {type(e).__name__}: {e})"]
        obs.open_dirs[d] = DirListing(path=d, entries=entries)

    for d in fields.get("CLOSEDIR", []):
        obs.open_dirs.pop(d, None)

    # FILES
    for fpath in fields.get("OPENFILE", []):
        p = Path(fpath)
        if not p.exists() or not p.is_file():
            obs.open_files[fpath] = FileSnapshot(fpath, ["(missing)"], [], 0, "")
            continue

        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="replace")
        raw_lines = text.splitlines()
        lines = [_truncate_line(ln, cfg.file_max_line_chars) for ln in raw_lines]

        head = lines[:cfg.file_head_lines]
        tail = lines[-cfg.file_tail_lines:] if len(lines) > cfg.file_head_lines else lines[cfg.file_head_lines:]
        trunc = max(0, len(lines) - (len(head) + len(tail)))

        obs.open_files[fpath] = FileSnapshot(fpath, head, tail, trunc, sha)

    for fpath in fields.get("CLOSEFILE", []):
        obs.open_files.pop(fpath, None)


# ----------------------------
# Program generation
# ----------------------------

def generate_program(
    cfg: RunnerConfig,
    history_text: str,
    observations_text: str,
    current_input: str,
    *,
    examples_jsonl: Optional[str],
    model: str,
    stream: bool,
) -> str:
    if not examples_jsonl:
        safe = repr(current_input)
        return "\n".join([
            "#!/usr/bin/env python3",
            f"print('stubbed model: input=' + {safe})",
            "print('<AIRESULT_STATUS>SUCCESS</AIRESULT_STATUS>')",
            "print('<AIRESULT_STEPS_REMAINING>0</AIRESULT_STEPS_REMAINING>')",
        ]) + "\n"

    task_input = history_text + "\n\n" + observations_text + "\n\n" + f"Input: {current_input}\n\nOutput:\n"

    out_text, _prompt_used, _resp_id = few_shot_openai_streaming(
        path_to_examples_jsonl=examples_jsonl,
        task_input=task_input,
        general_purpose=DEFAULT_GENERAL_PURPOSE,
        model=model,
        stream_to_console=stream,
    )
    return out_text


# ----------------------------
# Execution (interactive tee via shim)
# ----------------------------

SHIM_SOURCE = r'''
import argparse
import os
import runpy
import sys
from pathlib import Path

class Tee:
    def __init__(self, *targets):
        self.targets = targets
    def write(self, s):
        for t in self.targets:
            try:
                t.write(s)
            except Exception:
                pass
    def flush(self):
        for t in self.targets:
            try:
                t.flush()
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("stdout_log")
    ap.add_argument("stderr_log")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    prog = Path(args.program)
    out_path = Path(args.stdout_log)
    err_path = Path(args.stderr_log)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    out_f = out_path.open("w", encoding="utf-8", errors="replace", newline="")
    err_f = err_path.open("w", encoding="utf-8", errors="replace", newline="")

    sys.stdout = Tee(sys.__stdout__, out_f)
    sys.stderr = Tee(sys.__stderr__, err_f)

    if not args.interactive:
        sys.stdin = open(os.devnull, "r")

    code = 0
    try:
        runpy.run_path(str(prog), run_name="__main__")
    except SystemExit as e:
        if isinstance(e.code, int):
            code = e.code
        else:
            code = 0
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        try: out_f.flush()
        except Exception: pass
        try: err_f.flush()
        except Exception: pass
        try: out_f.close()
        except Exception: pass
        try: err_f.close()
        except Exception: pass

    raise SystemExit(code)

if __name__ == "__main__":
    main()
'''


def execute_script(cfg: RunnerConfig, script_text: str, run_dir: Path) -> Tuple[bool, int, int, str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    script_path = run_dir / "program.py"
    shim_path = run_dir / "shim.py"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    script_path.write_text(script_text, encoding="utf-8")
    shim_path.write_text(SHIM_SOURCE, encoding="utf-8")

    cmd = [cfg.python_exe, str(shim_path), str(script_path), str(stdout_path), str(stderr_path)]
    if cfg.interactive_child:
        cmd.append("--interactive")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cfg.workspace_root),
            timeout=cfg.timeout_seconds,
            check=False,
        )
        code = proc.returncode
        ok = (code == 0)
    except subprocess.TimeoutExpired:
        ok = False
        code = 124
        stderr_path.write_text("\n[runner] TIMEOUT\n", encoding="utf-8", errors="replace")
    except Exception as e:
        ok = False
        code = 125
        stderr_path.write_text(f"\n[runner] EXEC_ERROR: {type(e).__name__}: {e}\n", encoding="utf-8", errors="replace")

    dt_ms = int((time.time() - t0) * 1000)

    out_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    err_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return ok, code, dt_ms, out_text, err_text


# ----------------------------
# Session (multi-turn chat)
# ----------------------------

@dataclass
class Session:
    cfg: RunnerConfig
    examples_jsonl: Optional[str]
    model: str
    stream: bool

    records: List[RunRecord] = field(default_factory=list)
    obs: Observations = field(default_factory=Observations)

    def submit(self, user_input: str) -> RunRecord:
        current_input = user_input

        history_text = render_history(self.cfg, self.records)
        observations_text = self.obs.render()

        script = generate_program(
            cfg=self.cfg,
            history_text=history_text,
            observations_text=observations_text,
            current_input=current_input,
            examples_jsonl=self.examples_jsonl,
            model=self.model,
            stream=self.stream,
        )

        run_index = len(self.records) + 1
        run_dir = self.cfg.session_root / f"run_{run_index:04d}"

        ok, code, dt_ms, out_text, err_text = execute_script(self.cfg, script, run_dir)
        fields = extract_fields(out_text, err_text)

        rec = RunRecord(
            index=run_index,
            input_text=current_input,
            script_text=script,
            ok=ok,
            exit_code=code,
            duration_ms=dt_ms,
            fields=fields,
            stdout_render=render_stream(
                out_text,
                self.cfg.log_head_lines,
                self.cfg.log_tail_lines,
                self.cfg.max_line_chars,
                self.cfg.max_stream_chars,
            ),
            stderr_render=render_stream(
                err_text,
                self.cfg.log_head_lines,
                self.cfg.log_tail_lines,
                self.cfg.max_line_chars,
                self.cfg.max_stream_chars,
            ),
        )

        self.records.append(rec)
        apply_observation_side_effects(self.cfg, self.obs, fields)
        return rec



# ----------------------------
# CLI
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=None, help="Path to examples.jsonl for few-shot prompting.")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI model id.")
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming model output to console.")
    ap.add_argument("--workspace", default=None, help="Workspace root (default: cwd).")
    ap.add_argument("--session-dir", default=None, help="Session root for run artifacts (default: ./.runner_session).")

    # interactive child is ON by default; allow disabling
    ap.add_argument("--no-interactive-child", action="store_true", help="Disable child stdin (input() will EOF).")

    # optional: tune history size
    ap.add_argument("--max-stream-chars", type=int, default=None)
    ap.add_argument("--max-line-chars", type=int, default=None)
    ap.add_argument("--log-head-lines", type=int, default=None)
    ap.add_argument("--log-tail-lines", type=int, default=None)

    args = ap.parse_args(argv)

    cfg = RunnerConfig()
    if args.workspace is not None:
        cfg.workspace_root = Path(args.workspace).resolve()
    if args.session_dir is not None:
        cfg.session_root = Path(args.session_dir).resolve()
    cfg.session_root.mkdir(parents=True, exist_ok=True)

    if args.no_interactive_child:
        cfg.interactive_child = False

    if args.max_stream_chars is not None:
        cfg.max_stream_chars = args.max_stream_chars
    if args.max_line_chars is not None:
        cfg.max_line_chars = args.max_line_chars
    if args.log_head_lines is not None:
        cfg.log_head_lines = args.log_head_lines
    if args.log_tail_lines is not None:
        cfg.log_tail_lines = args.log_tail_lines

    s = Session(
        cfg=cfg,
        examples_jsonl=args.examples,
        model=args.model,
        stream=(not args.no_stream),
    )

    while True:
        try:
            line = input(">>> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if not line.strip():
            continue

        _ = s.submit(line.strip())
        print(render_history(cfg, s.records))
        print()
        print(s.obs.render())
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
