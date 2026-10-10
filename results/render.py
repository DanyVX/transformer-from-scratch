"""Render only measured tokenizer JSON records; do not supply manual numbers."""

import argparse
import json
from pathlib import Path


def render(path: Path) -> str:
    record = json.loads(path.read_text(encoding="utf-8"))
    environment = record["environment"]
    corpora = record["metrics"]["corpora"]
    lines = [
        "| Synthetic corpus | Tokens/sec | Tokens/character | Bytes/token |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(corpora):
        data = corpora[name]
        lines.append(
            f"| {name} | {data['tokens_per_second']:,.1f} | "
            f"{data['tokens_per_character']:.3f} | {data['bytes_per_token']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Measured source commit: {environment['git_commit']}",
            f"Runner: {environment['os']} / Python {environment['python']} "
            f"/ CPU count {environment['cpu_count']}; CUDA={environment['cuda_available']}.",
            "Synthetic corpus and a single short timing run. Do not extrapolate to large datasets.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(render(args.result))


if __name__ == "__main__":
    main()
