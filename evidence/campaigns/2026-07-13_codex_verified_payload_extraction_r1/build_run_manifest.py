from __future__ import annotations

import hashlib
import json
from pathlib import Path


CAMPAIGN_ID = "2026-07-13_codex_verified_payload_extraction_r1"
OPERATION_ID = "HRE_R1_VERIFIED_SCIENCE_EXTRACTION_20260713_CODEX_25"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

LOCKED_LOCAL = [
    "README.md",
    "build_run_manifest.py",
    "extractor.py",
    "independent_verify.py",
]

UPSTREAM = [
    "experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv/RUN_MANIFEST.json",
    "experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv/outputs/canonical_main/main_catalogs.jsonl",
    "experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv/outputs/canonical_main/main_results.json",
    "experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv/outputs/canonical_main/status.json",
    "experiments/2026-07-13_codex_static_verifier_topsis_gatefix/RUN_MANIFEST.json",
    "experiments/2026-07-13_codex_static_verifier_topsis_gatefix/outputs/canonical_overlay/FULL_VERIFICATION.json",
    "experiments/2026-07-13_codex_static_verifier_topsis_gatefix/outputs/canonical_overlay/verification_status.json",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/RUN_MANIFEST.json",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/FULL_VERIFICATION.json",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/STATUS.json",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/TERMINAL.json",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/VERIFICATION_STATUS.json",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/sealed_records.jsonl",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/RUN_MANIFEST.json",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/FULL_VERIFICATION.json",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/latency_results.json",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/raw_durations_ns.npy",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/status.json",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/timer_overhead_ns.npy",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/verification_status.json",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/RUN_MANIFEST.json",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/FULL_VERIFICATION.json",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/status.json",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/verification_status.json",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/xai_attributions.jsonl",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/xai_inputs.jsonl",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/xai_results.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_row(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    target = HERE / "RUN_MANIFEST.json"
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing manifest: {target}")
    manifest = {
        "schema_version": "hre.verified_science_extraction.run_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "LOCKED_BEFORE_EXTRACTION",
        "created_at": "2026-07-13T05:14:00+03:00",
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": OPERATION_ID,
        "locked_local_files": [file_row(HERE / rel, rel) for rel in LOCKED_LOCAL],
        "upstream_files": [file_row(ROOT / rel, rel) for rel in UPSTREAM],
        "output_directory": "outputs/canonical",
        "canonical_outputs": [
            "VERIFIED_SCIENTIFIC_PAYLOAD.json",
            "static_all_arms.csv",
            "static_locked_anchors.csv",
            "static_all_contrasts.csv",
            "static_anchor_diagnostics.csv",
            "lambda_posthoc_diagnostic.csv",
            "drift_all_conditions.csv",
            "latency_summary.json",
            "xai_run_level.jsonl",
            "xai_summary.json",
            "SOURCE_HASH_MANIFEST.json",
            "FULL_EXTRACTION_VERIFICATION.json",
        ],
    }
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"LOCKED {target} sha256={sha256(target)}")


if __name__ == "__main__":
    main()
