# pybrain

Minimal runner + few-shot OpenAI helper for "LLM writes an executable Python program" workflows.

## Contents

- `pybrain/brain.py`: interactive runner
- `pybrain/fewshot_openai.py`: few-shot prompt builder + OpenAI call helper (streaming)

## Install

```bash
pip install openai
```

Set your API key:

- macOS/Linux:
  ```bash
  export OPENAI_API_KEY="..."
  ```
- Windows (PowerShell):
  ```powershell
  setx OPENAI_API_KEY "..."
  ```

## Run (stub mode)

Stub mode does not call OpenAI. It generates a small program that emits:
- `<AIRESULT_STATUS>SUCCESS</AIRESULT_STATUS>`
- `<AIRESULT_STEPS_REMAINING>0</AIRESULT_STEPS_REMAINING>`

```bash
python -m pybrain.brain
```

## Run (few-shot OpenAI mode)

Provide a JSONL file where each line is:
`{"input":"...","output":"..."}`

```bash
python -m pybrain.brain --examples generate_program_examples.jsonl --model gpt-5-mini
```

Streaming is on by default (you see the program text as it's generated). Disable:

```bash
python -m pybrain.brain --examples generate_program_examples.jsonl --no-stream
```

## Protocol tags

The runner scans stdout+stderr for:

- `<AIRESULT_OPENGOAL>...</AIRESULT_OPENGOAL>` / `<AIRESULT_CLOSEGOAL>true</AIRESULT_CLOSEGOAL>`
- `<AIRESULT_OPENDIR>...</AIRESULT_OPENDIR>` / `<AIRESULT_CLOSEDIR>...</AIRESULT_CLOSEDIR>`
- `<AIRESULT_OPENFILE>...</AIRESULT_OPENFILE>` / `<AIRESULT_CLOSEFILE>...</AIRESULT_CLOSEFILE>`
- `<AIRESULT_STEPS_REMAINING>N</AIRESULT_STEPS_REMAINING>`

Semantics:
- `OPENGOAL`, `OPENDIR`, `OPENFILE` add to **Observations (active)** until closed.
- On success, if `STEPS_REMAINING > 0`, the runner reinvokes with `Input: "Continue"`.
- Full logs are always saved to `./.runner_session/run_XXXX/`.
