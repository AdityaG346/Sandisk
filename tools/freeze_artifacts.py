"""
tools/freeze_artifacts.py
=========================
Compute SHA-256 hashes of all frozen artifacts and write them to
outputs/ARTIFACT_HASHES.json.

Run this ONCE from the repo root to record the official hashes, then
run tools/check_frozen.py after every phase to detect any accidental
modifications.

Usage:
    python tools/freeze_artifacts.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FROZEN_FILES = [
    "outputs/model_a.pkl",
    "outputs/model_b.pkl",
    "outputs/model_a_meta.pkl",
    "outputs/model_b_meta.pkl",
    "outputs/predictions.csv",
    "outputs/comparison_table.csv",
    "outputs/per_die_explanations.md",
    "outputs/cache/test_probs_a.npy",
    "outputs/cache/test_probs_b.npy",
    "outputs/cache/blk_anom_test.npy",
    "outputs/cache/blk_test.parquet",
    "outputs/cache/blk_tr.parquet",
    "outputs/cache/blk_train_full.parquet",
    "outputs/cache/blk_val.parquet",
    "outputs/cache/die_anom_test.npy",
    "outputs/cache/pass_medians_a.parquet",
    "outputs/cache/pass_medians_b.parquet",
    "outputs/cache/sp_test.parquet",
    "outputs/cache/sp_tr.parquet",
    "outputs/cache/sp_val.parquet",
    "outputs/cache/test_meta.parquet",
]

OUTPUT_PATH = Path("outputs/ARTIFACT_HASHES.json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hashes: dict[str, str] = {}
    missing: list[str] = []

    for rel in FROZEN_FILES:
        p = repo_root / rel
        if not p.exists():
            print(f"  [MISSING] {rel}")
            missing.append(rel)
        else:
            digest = sha256_file(p)
            hashes[rel] = digest
            print(f"  [OK] {rel}  sha256={digest[:16]}...")

    if missing:
        print(f"\n[WARNING] {len(missing)} frozen file(s) not found — hashes not recorded for them.")

    out = repo_root / OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print(f"\n[freeze_artifacts] Wrote {len(hashes)} hashes to {out}")


if __name__ == "__main__":
    main()
