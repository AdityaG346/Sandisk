"""
test_part1_guardrails.py
========================
Top-level wrapper maintaining 100% backward compatibility with existing
documentation, test runners, and evaluation scripts.
The full regression suite is implemented in tests/test_guardrails.py.
"""
from tests.test_guardrails import (
    test_1_index_alignment_helper,
    test_2_training_sanity_check,
    test_3_threshold_independence,
    test_4_reproducibility,
    test_5_reload_from_disk_verification,
    main,
)

if __name__ == "__main__":
    main()
