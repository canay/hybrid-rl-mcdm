from __future__ import annotations

import verify_evidence


def test_manifest_and_hash_inventory() -> None:
    file_count, total_bytes = verify_evidence.verify_inventory()
    assert file_count > 100
    assert total_bytes > 80_000_000


def test_public_privacy_boundary() -> None:
    verify_evidence.verify_privacy_boundary()


def test_all_semantic_gates() -> None:
    gates = verify_evidence.verify_semantic_gates()
    assert gates["static_overlay"] == "completed_verified/PASS"
    assert gates["payload_extraction"] == "completed_verified/PASS"
