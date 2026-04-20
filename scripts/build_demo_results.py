"""
Generate data/demo_results.json — pre-cached ClassifiedReport outputs
for the 12 narratives in data/test_narratives.json.

Used by Streamlit Cloud (where Ollama is unavailable) to serve a working
demo without live inference.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEST_NARRATIVES_PATH, DEMO_RESULTS_PATH
from pipeline.classify import classify_report


def main():
    with open(TEST_NARRATIVES_PATH) as f:
        narratives = json.load(f)

    print(f"Building demo results for {len(narratives)} narratives...\n")

    demo_entries = []
    total = 0.0
    for idx, narrative in enumerate(narratives, start=1):
        print(f"[{idx:2d}/{len(narratives)}] {narrative[:70]}...")
        t0 = time.time()
        try:
            report = classify_report(narrative)
            elapsed = time.time() - t0
            total += elapsed
            demo_entries.append({
                "id": idx,
                "narrative": narrative,
                "report": report.model_dump(),
                "latency_seconds": round(elapsed, 2),
            })
            n_reactions = len(report.coded_reactions)
            print(
                f"    {'SERIOUS' if report.is_serious else 'non-serious':12s} "
                f"{n_reactions} reactions  ({elapsed:.1f}s)"
            )
        except Exception as e:
            elapsed = time.time() - t0
            total += elapsed
            print(f"    ERROR: {type(e).__name__}: {e}")
            demo_entries.append({
                "id": idx,
                "narrative": narrative,
                "error": f"{type(e).__name__}: {e}",
            })

    DEMO_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_RESULTS_PATH, "w") as f:
        json.dump(demo_entries, f, indent=2)

    print(f"\nSaved {len(demo_entries)} entries to {DEMO_RESULTS_PATH}")
    print(f"Total pipeline time: {total:.1f}s (avg {total/len(narratives):.1f}s per narrative)")


if __name__ == "__main__":
    main()
