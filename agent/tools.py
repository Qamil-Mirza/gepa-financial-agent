"""Tools: how agents get real data. OWNER: TODO

Right now this returns placeholder context. Once the team picks a data source,
implement it in data/sources.py and call it from here.
"""
from data import sources


def gather_context(task: str) -> str:
    return sources.fetch(task)
