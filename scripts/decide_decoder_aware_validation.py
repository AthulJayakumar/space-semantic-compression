"""Apply the frozen geographic-clustered selector advancement rule."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.decoder_aware_decision import advancement_decision, read_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    decision = advancement_decision(read_rows(args.results_dir / "matched_rate_rows.csv"))
    (args.results_dir / "advancement_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    with (args.results_dir / "geography_clustered_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision["comparisons"][0]))
        writer.writeheader()
        writer.writerows(decision["comparisons"])
    print(json.dumps({"decision": decision["decision"], "selected_candidate": decision["selected_candidate"]}, indent=2))


if __name__ == "__main__":
    main()
