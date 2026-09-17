from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproduction drivers for all manuscript figures."
    )
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--allow-large-fdm", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--output-dir", default="output/paper")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = Path(__file__).resolve().parent
    output = Path(arguments.output_dir)
    common = ["--preset", arguments.preset]
    if arguments.quiet:
        common.append("--quiet")

    commands = [
        [
            sys.executable,
            str(root / "reproduce_1d.py"),
            *common,
            "--output-dir",
            str(output / "figures01-06"),
        ],
        [
            sys.executable,
            str(root / "reproduce_2d.py"),
            *common,
            "--output-dir",
            str(output / "figures07-10"),
        ],
        [
            sys.executable,
            str(root / "reproduce_3d.py"),
            *common,
            "--output-dir",
            str(output / "figures11-15"),
        ],
    ]
    if arguments.allow_large_fdm:
        commands[1].append("--allow-large-fdm")
        commands[2].append("--allow-large-fdm")
    if arguments.save_data:
        commands[1].append("--save-data")
        commands[2].append("--save-data")

    if arguments.preset == "paper" and not arguments.allow_large_fdm:
        raise SystemExit(
            "Paper-scale FDM grids are intentionally guarded. Re-run with "
            "--allow-large-fdm on the documented HPC/GPU environment."
        )

    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
