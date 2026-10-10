"""Render measured scaling/ablation JSON into compact Markdown tables."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def render(path: Path) -> str:
    record = json.loads(path.read_text(encoding="utf-8"))
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for trial in record["metrics"]["trials"]:
        groups[str(trial["variant"])].append(trial)
    lines = [
        f"# Measured synthetic {record['metrics']['study']} study",
        "",
        "| Variant | Parameters | Val bits/byte mean | Seed sample std | Seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, trials in sorted(groups.items()):
        bpb = [float(row["bits_per_byte"]) for row in trials]
        std = statistics.stdev(bpb) if len(bpb) > 1 else float("nan")
        lines.append(
            f"| {label} | {int(trials[0]['params']):,} | "
            f"{statistics.mean(bpb):.4f} | {std:.4f} | {len(bpb)} |"
        )
    lines.append("")
    lines.append("Single small synthetic corpus; results are not statistically generalizable.")
    lines.append(
        f"Generated from committed JSON, source SHA {record['environment']['git_commit']}."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(render(args.result))


if __name__ == "__main__":
    main()
