"""Generate experimental plots from recorded results, never hand-enter numbers."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    payload = json.loads(args.record.read_text(encoding="utf-8"))
    trials = payload["metrics"]["trials"]
    x = [int(row["params"]) for row in trials]
    y = [float(row["bits_per_byte"]) for row in trials]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Held-out synthetic bits per byte")
    ax.set_title(payload["metrics"]["study"])
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
