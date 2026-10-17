"""Entry point. Usage:  python financial-agent/main.py "your goal here"

Command line for now. A web UI can call orchestrator.run() later.
"""
import sys

from agent import orchestrator

if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) or "Summarize the key financial risks for a small coffee shop."
    print("\n=== FINAL ANSWER ===\n" + orchestrator.run(goal))
