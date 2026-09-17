from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_release_files() -> list[str]:
    manifest = ROOT / "RELEASE_FILES.txt"
    entries = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def expand_release_files(entries: list[str]) -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []
    for entry in entries:
        source = ROOT / entry
        if source.is_file():
            files.append(source)
        elif source.is_dir():
            files.extend(
                path for path in sorted(source.rglob("*")) if path.is_file()
            )
        else:
            missing.append(entry)
    if missing:
        raise FileNotFoundError(
            "Release manifest contains missing entries: " + ", ".join(missing)
        )
    relative = [path.relative_to(ROOT) for path in files]
    if len(relative) != len(set(relative)):
        raise ValueError("Release manifest contains duplicate files")
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean, upload-ready SGIP source archive."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dist",
        help="directory in which to create the archive",
    )
    arguments = parser.parse_args()

    archive_root = "SGIP_code"
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "SGIP_code.zip"

    release_files = expand_release_files(read_release_files())

    with zipfile.ZipFile(
        archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for source in release_files:
            entry = source.relative_to(ROOT)
            bundle.write(source, f"{archive_root}/{entry.as_posix()}")

    checksum = sha256(archive)
    checksum_file = archive.with_suffix(".zip.sha256")
    checksum_file.write_text(
        f"{checksum}  {archive.name}\n", encoding="ascii"
    )

    print(f"Archive:  {archive}")
    print(f"SHA-256: {checksum}")
    print(f"Checksum: {checksum_file}")


if __name__ == "__main__":
    main()
