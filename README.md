# gepa-financial-agent

Built at the **Crusoe x Cerebral Valley Recursive Agents Hackathon** (Oct 17, 2026, SF).

> One-line pitch: _TODO_

## Team
| Name | GitHub | Focus |
|---|---|---|
| Qamil Mirza | @Qamil-Mirza | |
| Arjun Mittal | @Arjunmittal16 | |
| | | |
| | | |

## Quickstart
```bash
git clone https://github.com/Qamil-Mirza/gepa-financial-agent.git
cd gepa-financial-agent
python -m venv .venv && source .venv/bin/activate
pip install -r financial-agent/requirements.txt
cp .env.example .env          # then paste your Crusoe key into .env
python financial-agent/scripts/check_crusoe.py
```

## Run it
```bash
python financial-agent/main.py "your goal here"
```

## How it works
Planner breaks the goal into subtasks -> workers (sub-agents) do each one -> checker grades
each result and sends failures back with feedback -> synthesizer writes the final answer.
Every run is saved to `runs/`, and past checker feedback is fed to the planner next time.

| Role | Model |
|---|---|
| Planner, checker, synthesizer | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| Workers | `nvidia/Nemotron-3.5-Lightning-30B-A3B` |

## Layout (and who owns what)
```
financial-agent/
  main.py                  entry point (CLI)
  agent/
    orchestrator.py        the main loop            OWNER: TODO
    planner.py             goal -> subtasks         OWNER: TODO
    worker.py              does one subtask         OWNER: TODO
    checker.py             grades results           OWNER: TODO
    memory.py              saves runs, lessons      OWNER: TODO
    tools.py               how agents get data
    prompts.py             all system prompts (GEPA optimizes these)
    crusoe_client.py       shared Crusoe client, retries on 429/503
  data/sources.py          plug the real data source in here   OWNER: TODO
  gepa/                    prompt optimization, later
  scripts/check_crusoe.py  smoke test: key + model IDs
  docs/skills/             Crusoe hackathon skill file
```

## Crusoe notes
- Endpoint: `https://api.inference.crusoecloud.com/v1` (OpenAI-compatible)
- Rate limits are per model: 30 RPM without a payment method. Spread agent roles across models.
- See `financial-agent/docs/skills/` for model IDs, reasoning toggles, and troubleshooting.
