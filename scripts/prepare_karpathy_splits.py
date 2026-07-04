#!/usr/bin/env python3
"""Download and extract Karpathy COCO/Flickr30k split JSON files."""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
MEMBERS = ["dataset_coco.json", "dataset_flickr30k.json"]


def prepare_splits(output_dir: Path, url: str = DEFAULT_URL, force: bool = False) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "caption_datasets.zip"
    if force or not archive.exists():
        urllib.request.urlretrieve(url, archive)
    extracted: dict[str, str] = {"archive": str(archive), "url": url}
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for member in MEMBERS:
            if member not in names:
                raise FileNotFoundError(f"{member} not found in {archive}")
            target = output_dir / member
            if force or not target.exists():
                target.write_bytes(zf.read(member))
            # Validate JSON so failures happen before long extraction stages.
            json.loads(target.read_text(encoding="utf-8"))
            extracted[member] = str(target)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(extracted, indent=2, sort_keys=True), encoding="utf-8")
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_splits(args.output_dir, url=args.url, force=args.force), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
