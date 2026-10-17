# GEPA (later)

GEPA is a prompt optimizer in the DSPy library. Plan: run the agent on a set of example goals,
score the results, and let GEPA rewrite the prompts in `agent/prompts.py` to do better.

TODO(team):
1. Collect 10-20 example goals with what a good answer looks like.
2. Write a scoring function (the checker agent could be a starting point).
3. `pip install dspy` and wire up GEPA against `prompts.py`.
