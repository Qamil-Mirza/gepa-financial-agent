# GEPA demo UI: design

Date: 2026-10-17. Branch: `ui-improvements`.

## Intent

A one-minute, click-through demo that makes one point: **simply rewriting the supervisor's prompt (GEPA) made a
small model better at catching drift, and we only ran a handful of loops. Imagine more.** The prompt is the hero.
Everything before the prompt screen is setup.

The audience should leave saying: "something as simple as rewriting the prompt actually worked. The gains weren't
huge, but they only ran a few loops. What if they ran more?"

The main constraint from the user: **the UI must not be cluttered.** One idea per screen, readable from the back
of a room in a few seconds.

## Goals

- Tell the story in four screens that a presenter clicks through in about 60 seconds.
- Show a baseline supervisor and a GEPA-optimized supervisor watching the **same** agents, side by side, and
  show which one flags the drift (and when).
- Show at a glance that the prompt changed across GEPA iterations while the score went up.
- End on a scoreboard: Nano (original prompt) vs Nano + GEPA vs Super (original prompt).
- Two modes: **demo mode** (replay of a real recorded run, used for the one-minute slot) and **live mode**
  (really spawns agents and supervisors on Crusoe, a backup for questions afterwards).

## Non-goals

- Not a general debugging dashboard. The existing PitStop "pit wall" (`pitstop/pitstop/pitstop/ui/`) stays as it
  is for that.
- No readable prompt text or diffs on the main prompt screen. The audience reads the score, not the prompt.
- No DAG layout. The three agents run in parallel and do not depend on each other (`asyncio.gather` in
  `pitstop/pitstop/pitstop/orchestrator.py`), so they are drawn as three lanes. (Our call; see below.)
- No Super + GEPA bar on the scoreboard, even though the evaluation chain produces that number. Three bars keep it
  uncluttered. (Our call.)

## Anti-goals (failure even if it technically works)

- **Scripted or invented results.** Demo mode replays a real recorded run and real evaluation numbers. If the
  recording shows GEPA flagging drift at the same step as the baseline, the demo shows that. If Nano + GEPA does
  not beat Nano on test, the bars show that. Every number on screen must be traceable to a file.
- **Unlabeled projection.** The "what if we ran more loops" extension on the score chart must be visibly dashed and
  labeled as not measured.
- **Clutter.** If a screen needs explaining, it has too much on it.
- **Unfair comparison.** The two supervisors must judge identical agent behavior, not two swarms that happened to
  behave differently.

## Models and arms

- **Nano:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`. The baseline model.
- **Super:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` ("nemotron-v3-super").
- **Original prompt:** `pitstop/pitstop/prompts/supervisor_v0.txt` (`supervisor_v0@d7b64192`).
- **GEPA prompt:** the best candidate from the Nano GEPA run `runs/gepa/pitstop_nano_800/` (in progress on
  2026-10-17), saved by `scripts/optimize_pitstop.py` to `pitstop/pitstop/prompts/gepa/pitstop_nano_800.txt`.

Left pane = Nano + original prompt. Right pane = Nano + GEPA prompt. Same model on both sides, so the prompt is the
only difference.

## The four screens

Each click advances one screen. A presenter talks over it. Approximate times are for a 60-second slot.

### 1. Start (about 5 s)

- A single chat box, centered, nothing else on screen except the product name.
- The presenter types (or the box is prefilled with) the swarm objective and presses enter.
- Visual feedback that three agents are spawning (for example, three agent chips appear and slide toward the next
  screen).

### 2. Watch (about 20 s)

- Two panes side by side. Left: "Nano, original prompt". Right: "Nano + GEPA prompt".
- Each pane: one supervisor at the top, three agent lanes beneath it (NVIDIA data center, AMD AI, supply chain).
  The supervisor is drawn as watching all three lanes.
- Each lane advances step by step. Each supervisor check is a colored cell: on track / uncertain / drifting,
  reusing the pit wall's color language.
- The key beat: drift is injected into the NVIDIA lane at step 2 (it wanders into consumer gaming GPUs). When a
  supervisor flags it, that pane's NVIDIA lane lights up with a clear "Drift caught at step N" marker. If one pane
  flags earlier than the other, that difference must be obvious without reading anything.
- Small, unobtrusive ground-truth marker in the NVIDIA lane: "drift started here" at step 2.
- Demo mode replays faster than real time so the whole screen fits in about 20 seconds; the replay starts when the
  presenter lands on the screen.
- When the agents finish, each lane collapses into a one-line result (the gist of that agent's final report). There
  is no separate outputs screen.

### 3. Prompt evolution (about 20 s): the hero

- **Main element:** a large score line, one point per GEPA iteration, rising left to right. This is what the eye
  lands on first.
- **Supporting element:** beside it, a small "minimap" of the prompt text, too small to read. As each accepted
  iteration appears, its added or changed text appears as highlighted blocks in the minimap and the minimap grows,
  so at a glance the prompt is visibly changing. Rejected attempts may flash briefly and disappear (builder's call
  whether this helps or clutters).
- **Ending:** after the last real point, a short dashed extension labeled as not measured ("what if we ran more
  loops?"). The caption names the real number of loops from the run's data, not a hard-coded "5".

### 4. Scoreboard (about 15 s)

- Three bars: Nano (original prompt), Nano + GEPA, Super (original prompt).
- Headline metric: **accuracy on the held-out test split** of PitStop's hard-pairs set, as a percentage. Accuracy
  covers both halves of the job: firing on drift and staying quiet on healthy agents.
- Under each bar, in small text: drift caught (drift recall) and false alarms.
- Nothing else on the screen.

## Modes

**Demo mode** (default for the presentation). Replays a recorded run end to end with no network calls and no
rate-limit risk. Each screen reads from a data snapshot:

- Screen 2: a recorded swarm run in which both supervisors judged every identical supervisor input.
- Screen 3: the GEPA run's candidates, lineage and per-iteration scores.
- Screen 4: the test-split evaluation results for the three arms.

**Live mode** (backup). The chat box starts a real run on Crusoe: one swarm of three agents, with both supervisors
(Nano + original, Nano + GEPA) judging every identical input as it happens. Screens 3 and 4 show the same
recorded GEPA and evaluation data as demo mode (they are offline results by nature). The Crusoe account is capped
at 30 requests/minute and has hit 429s, so live mode must degrade visibly but gracefully (show "waiting on model"
rather than freezing or crashing).

A switch picks the mode (where it lives is the builder's call; it should not be visible clutter during the demo).

## Data that exists today (2026-10-17)

- **GEPA run (screen 3):** `runs/gepa/pitstop_nano_800/` (gitignored `runs/`). `candidates.json` holds every
  candidate's full prompt text. `candidate_tree.html` embeds a `NODES` array with each candidate's `idx`, val
  `score` and `parents`. `run_log.json` / `run_log.txt` have per-iteration selection, minibatch scores and
  accept/reject. The run finished at 15:27 on 2026-10-17: 15 iterations, 9 candidates (seed + 8 accepted), val
  0.800 (seed) → best 0.854. The best prompt's lineage is seed → 2 → 3 → 8 (0.800 → 0.811 → 0.829 → 0.854),
  growing from 1.6k to 7.8k characters. Accepted candidates are a tree (parents 0, 0, 2, 1, 3, 1, 3, 3), and
  "accepted" only means it passed the minibatch check: candidates 4 (0.765) and 6 (0.794) scored below the seed.
  The best prompt is saved at `pitstop/pitstop/prompts/gepa/pitstop_nano_800.txt`.
- **Test results (screen 4):** produced automatically when each GEPA run finishes. The chain runs
  `python -m evals.run_hard_cases --splits test --trials 5` for each arm and writes to
  `pitstop/pitstop/runs/compare_nano/` (arms: `supervisor_v0`, `pitstop_nano_800`, `gepa_nano_150`) and
  `pitstop/pitstop/runs/compare_super/` (arms: `supervisor_v0`, `pitstop_super_600`, `gepa_team_300`).
  `scripts/summarize_pitstop.py` turns these into accuracy / drift recall / false alarms. Earlier baseline numbers
  (3 trials, `pitstop/pitstop/evals/results/baseline_v0_{nano,super}.json`): test accuracy Nano 0.847, Super 0.889;
  drift recall Nano 0.694, Super 0.917.
- **Swarm recording (screen 2): does not exist yet.** `pitstop/pitstop/runs/` is empty. One live run of
  `pitstop/pitstop/scenarios/nvda_amd.json` is needed, with both supervisors judging each input. The orchestrator
  currently has one supervisor in the loop; `python -m pitstop.cli compare` can re-judge a recorded run's exact
  inputs with a second prompt, without new research calls. That run costs You.com calls (about $0.11 each) plus
  Crusoe calls.

`runs/` is gitignored, so the demo must not depend on a teammate's local `runs/`: the data it replays needs to be
exported into a committed snapshot.

## Decided by the user

- Demo mode and live mode both exist; the one-minute slot uses demo mode.
- One minute total, presenter clicks through.
- Four screens, in the order above. Agent outputs fold into the end of the watch screen.
- Prompt screen: score going up is the focus; the prompt changing is visible at a glance, not read.
- Scoreboard headline: test accuracy, with drift recall and false alarms in small text.
- One set of agents, two supervisors watching it.

## Settled after the design review (our calls)

- **"Drift caught at step N"** uses the eval/GEPA firing rule: the first DRIFTING verdict at confidence ≥ 0.75.
  That is the rule GEPA optimized for and the rule screen 4's accuracy is built on, so screens 2 and 4 agree.
  (The orchestrator's alert policy also fires on two DRIFTING verdicts in a row at ≥ 0.5; the demo does not use it.)
- **Score line:** best-so-far val score over all iterations (x axis = iteration, 0 to 15), labeled "best prompt so
  far". Accepted candidates that scored lower appear as faint dots; rejected iterations as faint marks on the axis.
  The minimap shows the best-so-far prompt, highlighting lines not in its parent.
- **"Loops" in the caption** is the real iteration count from the run (15 for `pitstop_nano_800`), not "5".
- **Live mode always runs the `nvda_amd` scenario.** The chat box is prefilled with its objective; typed text
  replaces only the swarm objective. Turning free text into three agent directives is out of scope.
- **Live research is real** (paid You.com calls through PitStop's existing client and cache), capped at the sum of
  the agents' `max_steps` (14). Expect roughly 5–10 minutes and about $1.50 per live run. It is a backup for
  questions, not the one-minute slot.
- **Recording for screen 2:** one live run of `nvda_amd` with both supervisors. Re-record only for infrastructure
  reasons (429s or transport failures in the record, research cap hit, or the injected drift never actually
  happening). Never re-record to get a better-looking outcome; screen 2 must handle "both caught it at the same
  step", "baseline caught it first" and false alarms on healthy lanes.
- **Super bar:** from `pitstop/pitstop/runs/compare_super/` (5 trials) if it exists at export time; otherwise
  `pitstop/pitstop/evals/results/baseline_v0_super.json`, labeled on screen as an earlier 3-trial run. The export
  script can be re-run any time before the demo.

## Our calls (the user can override)

- Three lanes instead of a DAG (agents are independent).
- No Super + GEPA bar.
- Replay starts automatically on landing on screen 2.
- Dashed, labeled extension at the end of the score line.
- Visual language reuses the existing pit wall (dark asphalt theme, green / yellow / red status colors, Barlow
  fonts) so it looks like part of the same product.

## Left to the builder

- Tech: extend the existing stdlib server and single-page approach, or something else, as long as demo mode runs
  offline from committed data with one command.
- Exact animations, replay speed, layout proportions.
- How "added text" is computed for the minimap (diff against parent candidate vs against seed).
- Whether the score line plots each accepted candidate's val score or best-so-far (must be labeled honestly either
  way; note the val score can go down, e.g. iteration 10 scored 0.765).
- Snapshot format and where it is committed.
- How live mode streams to the browser (the existing `pitstop/pitstop/pitstop/server.py` already has SSE timeline
  streaming).

## Changes after the first build (2026-10-17, user decisions)

- **Correction mode (Gautam's `--correct-on-drift`, merged from `gautam-bh/feature`).** The GEPA supervisor is
  in charge of all three agents: at an intervention point it blocks the drifting query and sends a correction, and
  the agent re-plans that step. The original prompt judges the exact same inputs as a shadow ("watching only") and
  cannot act. Pane layout comes from `demo.json` in the run dir (`{"left": "shadow", "right": "primary"}`).
- **Re-plans after a correction** appear as a "↩" cell right after the flagged step, in both panes (same input);
  the pane in charge shows a "corrected" marker and "corrected → back on track" (or "still drifting").
- **"Why?" panel.** Clicking any checked cell opens the supervisor's verdict, reasoning, cited evidence, the
  agent's query and any correction, from the curated run timeline (`pitstop/timeline.py`), built for both
  supervisors.
- **Failed checks** (rate-limited call or non-JSON reply) render as "no verdict", never as UNCERTAIN.
- **False alarms** (fires on a lane with no injected drift) are counted per pane and in the closing verdict.
- **Live mode is a first-class path for the demo:** worker at temperature 0 (a rehearsal warms the You.com cache),
  6 LLM retries on 429s, page re-attaches to a running live run after a reload. A live run takes about 4–6 minutes;
  the catch and correction land about 1.5 minutes in.
- **Text sized for a projector**, and a Home button / Home or Esc key back to the start screen.
