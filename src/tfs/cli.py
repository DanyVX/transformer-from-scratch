"""Small cross-platform CLI for the setup milestone."""

import argparse
import json
from collections.abc import Sequence
from typing import Any

from tfs.config import PROFILE_NAMES, RuntimeSettings, load_profile
from tfs.provenance import environment_metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tfs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profiles", help="List named model configurations")
    show = sub.add_parser("config", help="Show a validated named configuration")
    show.add_argument("--profile", choices=PROFILE_NAMES)
    sub.add_parser("environment", help="Print non-sensitive runtime metadata")
    args = parser.parse_args(argv)

    data: dict[str, Any]
    if args.command == "profiles":
        data = {"profiles": list(PROFILE_NAMES)}
    elif args.command == "config":
        profile = args.profile or RuntimeSettings().profile
        data = load_profile(profile).model_dump(mode="json")
        data["predicted_parameter_count"] = load_profile(profile).model.parameter_count()
        data["parameter_count_measured"] = False
    else:
        data = environment_metadata()
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0
