#!/usr/bin/env python3
# pybrain.brain - minimal, debuggable request-fulfillment runner.
#
# - Maintains History: prior Inputs, preview of generated script, stdout/stderr head+tail.
# - Maintains Observations (active): OPENGOAL/OPENDIR/OPENFILE until closed.
# - Invokes a program generator (stub or OpenAI few-shot).
# - Executes generated Python program; logs to .runner_session/run_XXXX/.
# - Auto-continues with Input: "Continue" while AIRESULT_STEPS_REMAINING > 0 (on success).
#
# AIRESULT tags scanned from stdout+stderr:
#   <AIRESULT_NAME>value</AIRESULT_NAME>

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .fewshot_openai import few_shot_openai_streaming


@dataclass
class RunnerConfig:
    workspace_root: Path = field(default_factory=Path.cwd)
    session_root: Path = field(default_factory=lambda: Path.cwd() / ".runner_session")
    python_exe: str = sys.executable

    log_head_lines: int = 40
    log_tail_lines: int = 40

    script_preview_head: int = 3
    script_preview_tail: int = 2

    file_head_lines: int = 60
    file_tail_lines: int = 60

    max_auto_steps: int = 12
    timeout_seconds: Optional[int] = None


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


def _split_head_tail(lines: List[str], head_n: int, tail_n: int) -> Tuple[List[str], List[str], int]:
    if len(lines) <= head_n + tail_n:
        return lines, [], 0
    head = lines[:head_n]
    tail = lines[-tail_n:]
    trunc = len(lines) - (len(head) + len(tail))
    return head, tail, trunc


def render_stream(text: str, head_n: int, tail_n: int) -> str:
    if text == "":
        return "      (empty)"
    lines = text.splitlines()
    head, tail, trunc = _split_head_tail(lines, head_n, tail_n)
    out = [f"      {ln}" for ln in head]
    if trunc > 0:
        out.append(f"      [ {trunc} lines truncated ]")
        out.extend([f"      {ln}" for ln in tail])
    return "\n".join(out)


def render_script_preview(script: str, head_n: int, tail_n: int) -> str:
    lines = script.splitlines()
    if not lines:
        return "      (empty)"
    if len(lines) <= head_n + tail_n:
        return "\n".join([f"      {ln}" for ln in lines])
    head = lines[:head_n]
    tail = lines[-tail_n:]
    trunc = len(lines) - (len(head) + len(tail))
    out = [f"      {ln}" for ln in head]
    out.append(f"      [ truncated {trunc} lines ]")
    out.extend([f"      {ln}" for ln in tail])
    return "\n".join(out)


def render_history(records: List[RunRecord]) -> str:
    if not records:
        return "History: (none)"
    out = ["History (most recent last):", ""]
    for r in records:
        out.append(f"[{r.index}] Input: {r.input_text!r}")
        out.append("    Output:")
        out.append(render_script_preview(r.script_text, 3, 2))
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
    if "OPENGOAL" in fields:
        obs.active_goal = fields["OPENGOAL"][-1]
    if _get_bool_field(fields, "CLOSEGOAL"):
        obs.active_goal = None

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

    for fpath in fields.get("OPENFILE", []):
        p = Path(fpath)
        if not p.exists() or not p.is_file():
            obs.open_files[fpath] = FileSnapshot(fpath, ["(missing)"], [], 0, "")
            continue
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        head = lines[:cfg.file_head_lines]
        tail = lines[-cfg.file_tail_lines:] if len(lines) > cfg.file_head_lines else lines[cfg.file_head_lines:]
        trunc = max(0, len(lines) - (len(head) + len(tail)))
        obs.open_files[fpath] = FileSnapshot(fpath, head, tail, trunc, sha)

    for fpath in fields.get("CLOSEFILE", []):
        obs.open_files.pop(fpath, None)


def generate_program(
    history_text: str,
    observations_text: str,
    current_input: str,
    *,
    examples_jsonl: Optional[str],
    general_purpose: str,
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
        general_purpose=general_purpose,
        model=model,
        stream_to_console=stream,
    )
    return out_text


def execute_script(cfg: RunnerConfig, script_text: str, run_dir: Path) -> Tuple[bool, int, int, str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    script_path = run_dir / "program.py"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    script_path.write_text(script_text, encoding="utf-8")

    t0 = time.time()
    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        try:
            proc = subprocess.run(
                [cfg.python_exe, str(script_path)],
                cwd=str(cfg.workspace_root),
                stdout=out_f,
                stderr=err_f,
                timeout=cfg.timeout_seconds,
                check=False,
            )
            code = proc.returncode
            ok = (code == 0)
        except subprocess.TimeoutExpired:
            ok = False
            code = 124
            err_f.write(b"\n[runner] TIMEOUT\n")
        except Exception as e:
            ok = False
            code = 125
            err_f.write(f"\n[runner] EXEC_ERROR: {type(e).__name__}: {e}\n".encode("utf-8", errors="replace"))

    dt_ms = int((time.time() - t0) * 1000)
    out_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    err_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    return ok, code, dt_ms, out_text, err_text


@dataclass
class Session:
    cfg: RunnerConfig
    examples_jsonl: Optional[str]
    general_purpose: str
    model: str
    stream: bool

    records: List[RunRecord] = field(default_factory=list)
    obs: Observations = field(default_factory=Observations)

    def submit(self, user_input: str) -> RunRecord:
        current_input = user_input
        auto = 0
        last_rec: Optional[RunRecord] = None

        while True:
            history_text = render_history(self.records)
            observations_text = self.obs.render()

            script = generate_program(
                history_text=history_text,
                observations_text=observations_text,
                current_input=current_input,
                examples_jsonl=self.examples_jsonl,
                general_purpose=self.general_purpose,
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
                stdout_render=render_stream(out_text, self.cfg.log_head_lines, self.cfg.log_tail_lines),
                stderr_render=render_stream(err_text, self.cfg.log_head_lines, self.cfg.log_tail_lines),
            )
            self.records.append(rec)
            apply_observation_side_effects(self.cfg, self.obs, fields)
            last_rec = rec

            if not ok:
                return last_rec

            steps_remaining = _get_int_field(fields, "STEPS_REMAINING", 0)
            if steps_remaining > 0:
                auto += 1
                if auto >= self.cfg.max_auto_steps:
                    return last_rec
                current_input = "Continue"
                continue

            return last_rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=None, help="Path to examples.jsonl for few-shot prompting.")
    ap.add_argument("--purpose", default="You are an agent that completes the Output by writing a single executable Python program.",
                    help="Short description of the LLM's general purpose.")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI model id.")
    ap.add_argument("--no-stream", action="store_true", help="Disable streaming model output to console.")
    ap.add_argument("--workspace", default=None, help="Workspace root (default: cwd).")
    ap.add_argument("--session-dir", default=None, help="Session root for run artifacts (default: ./.runner_session).")
    args = ap.parse_args(argv)

    cfg = RunnerConfig()
    if args.workspace is not None:
        cfg.workspace_root = Path(args.workspace).resolve()
    if args.session_dir is not None:
        cfg.session_root = Path(args.session_dir).resolve()
    cfg.session_root.mkdir(parents=True, exist_ok=True)

    s = Session(
        cfg=cfg,
        examples_jsonl=args.examples,
        general_purpose=args.purpose,
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
        print(render_history(s.records))
        print()
        print(s.obs.render())
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
