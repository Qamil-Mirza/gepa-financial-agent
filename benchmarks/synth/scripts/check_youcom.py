"""Make one real You.com Finance Research request and save the raw response for inspection.

Costs one 'deep' request (~$0.11) and can take up to 120s.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from synth import config

query = " ".join(sys.argv[1:]) or "NVIDIA data center segment revenue for the last four reported quarters"
key = config.youcom_key()
if not key:
    sys.exit("YDC_API_KEY not set")

t0 = time.time()
resp = httpx.post(
    config.YOUCOM_URL,
    headers={"X-API-Key": key},
    json={"input": query, "research_effort": "deep"},
    timeout=180,
)
elapsed = time.time() - t0
print(f"HTTP {resp.status_code} in {elapsed:.1f}s")
resp.raise_for_status()
body = resp.json()
out = config.ROOT / "evidence" / "youcom_sample_response.json"
out.write_text(json.dumps({"query": query, "elapsed_s": round(elapsed, 1), "response": body}, indent=2))
print("saved", out)
print("top-level keys:", list(body))
output = body.get("output", {})
print("output keys:", list(output))
print("content preview:", str(output.get("content"))[:400])
for s in output.get("sources", [])[:3]:
    print("source keys:", list(s), "|", s.get("title"))
print("num sources:", len(output.get("sources", [])))
