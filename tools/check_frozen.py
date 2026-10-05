"""
tools/check_frozen.py
=====================
Verify that all frozen artifacts match the SHA-256 hashes recorded in
outputs/ARTIFACT_HASHES.json.

Usage:
    python tools/check_frozen.py

Returns exit-code 0 if all hashes match, 1 if any mismatch is detected.
Run after EVERY phase.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HASH_FILE = Path("outputs/ARTIFACT_HASHES.json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hash_path = repo_root / HASH_FILE

    if not hash_path.exists():
        print(f"[check_frozen] HASH FILE NOT FOUND: {hash_path}")
        print("  Run `python tools/freeze_artifacts.py` first.")
        return 1

    with open(hash_path, encoding="utf-8") as f:
        expected: dict[str, str] = json.load(f)

    n_ok = 0
    n_fail = 0

    for rel, expected_digest in expected.items():
        p = repo_root / rel
        if not p.exists():
            print(f"  [MISSING] {rel}")
            n_fail += 1
            continue
        actual = sha256_file(p)
        if actual == expected_digest:
            n_ok += 1
        else:
            print(f"  [MISMATCH] {rel}")
            print(f"    expected: {expected_digest}")
            print(f"    actual:   {actual}")
            n_fail += 1

    if n_fail == 0:
        print(f"[check_frozen] ALL {n_ok} frozen artifacts OK.")
        return 0
    else:
        print(f"\n[check_frozen] FAILED: {n_fail} artifact(s) modified or missing!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
