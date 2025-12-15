# pybrain (v0.2)

Runner + few-shot OpenAI helper for "LLM writes an executable Python program" workflows.

## Changes in v0.2

- Interactive child execution is **ON by default** (stdin inherited).
- Child stdout/stderr are **mirrored to the console** while also being written to log files (via a shim).
- History truncation is bounded by **lines and characters** (prevents huge histories).

## Run

Stub mode:

```bash
python -m pybrain.brain
```

Few-shot mode:

```bash
python -m pybrain.brain --examples generate_program_examples.jsonl --model gpt-5-mini
```

Disable streaming model output:

```bash
python -m pybrain.brain --examples generate_program_examples.jsonl --no-stream
```

Disable interactive child stdin:

```bash
python -m pybrain.brain --no-interactive-child
```

## Artifacts

Logs and executed programs are stored under:

`./.runner_session/run_XXXX/`

## AIRESULT protocol

The runner scans stdout+stderr for tags like:

- `<AIRESULT_OPENGOAL>...</AIRESULT_OPENGOAL>` / `<AIRESULT_CLOSEGOAL>true</AIRESULT_CLOSEGOAL>`
- `<AIRESULT_OPENDIR>...</AIRESULT_OPENDIR>` / `<AIRESULT_CLOSEDIR>...</AIRESULT_CLOSEDIR>`
- `<AIRESULT_OPENFILE>...</AIRESULT_OPENFILE>` / `<AIRESULT_CLOSEFILE>...</AIRESULT_CLOSEFILE>`
- `<AIRESULT_STEPS_REMAINING>N</AIRESULT_STEPS_REMAINING>`

These update **Observations (active)** until closed, and drive multi-step `Continue` runs.
