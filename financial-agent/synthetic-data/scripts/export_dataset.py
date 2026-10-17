"""Re-verify every stored trace, then export labeled prefix examples to data/.

Usage: python scripts/export_dataset.py [--out data]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import config
from synth.export import export
from synth.store import TraceStore
from synth.verify import verify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.DATA_DIR)
    args = ap.parse_args()
    traces = TraceStore().all()
    for t in traces:  # verifier rules may have changed since generation
        t.generation.verification = verify(t)
    counts = export(traces, args.out)
    passed = sum(t.generation.verification.passed for t in traces)
    print(f"{passed}/{len(traces)} traces verified; examples per split: {counts}; written to {args.out}")


if __name__ == "__main__":
    main()
