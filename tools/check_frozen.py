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


def verify_frozen(repo_root: Path | None = None) -> tuple[bool, dict[str, str]]:
    """Verify all frozen artifacts. Returns (all_ok, results_dict)."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    hash_path = repo_root / HASH_FILE

    if not hash_path.exists():
        return False, {"HASH_FILE": "NOT FOUND"}

    with open(hash_path, encoding="utf-8") as f:
        expected: dict[str, str] = json.load(f)

    results: dict[str, str] = {}
    all_ok = True

    for rel, expected_digest in expected.items():
        p = repo_root / rel
        if not p.exists():
            results[rel] = "MISSING"
            all_ok = False
            continue
        actual = sha256_file(p)
        if actual == expected_digest:
            results[rel] = "OK"
        else:
            results[rel] = f"MISMATCH (expected {expected_digest[:8]}, got {actual[:8]})"
            all_ok = False

    return all_ok, results


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hash_path = repo_root / HASH_FILE

    if not hash_path.exists():
        print(f"[check_frozen] HASH FILE NOT FOUND: {hash_path}")
        print("  Run `python tools/freeze_artifacts.py` first.")
        return 1

    all_ok, results = verify_frozen(repo_root)

    n_ok = sum(1 for s in results.values() if s == "OK")
    n_fail = len(results) - n_ok

    for rel, status in results.items():
        if status != "OK":
            print(f"  [{status}] {rel}")

    if all_ok:
        print(f"[check_frozen] ALL {n_ok} frozen artifacts OK.")
        return 0
    else:
        print(f"\n[check_frozen] FAILED: {n_fail} artifact(s) modified or missing!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
