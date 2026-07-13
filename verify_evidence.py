#!/usr/bin/env python3
"""Verify file integrity, privacy boundaries, and key scientific gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "EVIDENCE_MANIFEST.json"
SUMS_PATH = ROOT / "SHA256SUMS.txt"
EXCLUDED_FILES = {MANIFEST_PATH.name, SUMS_PATH.name}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def actual_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.name in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files[relative.as_posix()] = path
    return files


def verify_inventory() -> tuple[int, int]:
    manifest = load_json(MANIFEST_PATH.name)
    require(manifest.get("schema_version") == "hre.public_evidence_manifest.v1", "manifest schema")
    rows = manifest.get("files")
    require(isinstance(rows, list), "manifest files list")
    expected = {row["path"]: row for row in rows}
    require(len(expected) == len(rows), "duplicate manifest path")
    actual = actual_files()
    require(set(expected) == set(actual), f"inventory mismatch: missing={sorted(set(expected)-set(actual))}, unexpected={sorted(set(actual)-set(expected))}")

    for relative, row in expected.items():
        path = actual[relative]
        require(path.stat().st_size == row["bytes"], f"size mismatch: {relative}")
        require(sha256(path) == str(row["sha256"]).lower(), f"hash mismatch: {relative}")

    require(manifest.get("file_count") == len(rows), "manifest file_count")
    require(manifest.get("total_bytes") == sum(int(row["bytes"]) for row in rows), "manifest total_bytes")

    sums: dict[str, str] = {}
    for line in SUMS_PATH.read_text(encoding="utf-8-sig").splitlines():
        digest, relative = line.split("  ", 1)
        sums[relative] = digest
    require(sums == {path: row["sha256"] for path, row in expected.items()}, "SHA256SUMS mismatch")
    return len(rows), int(manifest["total_bytes"])


def verify_privacy_boundary() -> None:
    forbidden_parts = {"archive", "render", "tmp", "MD", "_state", "safe_data", "__pycache__", ".pytest_cache"}
    forbidden_names = {"users.json", "items.csv"}
    text_suffixes = {".md", ".py", ".json", ".jsonl", ".csv", ".txt", ".log", ".cff"}
    # Build these markers from fragments so the verifier does not flag its own
    # source merely for defining the privacy rule.
    local_markers = (
        b"C:" + b"\\DOCS\\AKADEMIK",
        b"C:" + b"/DOCS/AKADEMIK",
        b"/home" + b"/ubuntu/",
    )
    secret_signature_re = re.compile(
        b"(?i)("
        + b"AK"
        + b"IA[0-9A-Z]{16}|"
        + b"gh"
        + b"[pousr]_[A-Za-z0-9]{20,}|"
        + b"-----BEGIN "
        + b"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
    forbidden_secret_names = {
        ".env",
        ".env.local",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
    markdown_link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

    for relative, path in actual_files().items():
        parts = Path(relative).parts
        require(not any(part in forbidden_parts for part in parts), f"forbidden path part: {relative}")
        require(path.name.lower() not in forbidden_names, f"forbidden raw-data filename: {relative}")
        require(path.name.lower() not in forbidden_secret_names, f"forbidden secret filename: {relative}")
        require(path.stat().st_size < 100 * 1024 * 1024, f"file at or above 100 MiB: {relative}")
        if path.suffix.lower() == ".csv" and "processed" in (part.lower() for part in parts):
            raise EvidenceError(f"processed source CSV is forbidden: {relative}")
        if path.suffix.lower() in text_suffixes:
            data = path.read_bytes()
            require(not any(marker in data for marker in local_markers), f"absolute local path marker: {relative}")
            require(secret_signature_re.search(data) is None, f"secret signature: {relative}")
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8-sig")
            for raw_target in markdown_link_re.findall(text):
                target = raw_target.strip().strip("<>").split("#", 1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:", "doi:")):
                    continue
                require((path.parent / target).resolve().exists(), f"broken Markdown link: {relative} -> {target}")

    scan_report = load_json("evidence/privacy/CURRENT_PUBLIC_SCAN.json")
    require(scan_report["schema_version"] == "hre.current_public_repository_scan.v1", "current public scan schema")
    require(scan_report["verdict"] == "PASS", "current public scan verdict")
    require(scan_report["scanned_files"] == len(actual_files()), "current public scan file count")
    require(all(int(value) == 0 for value in scan_report["checks"].values()), "current public scan checks")


def verify_primary_profile_boundary(catalogs_path: Path, primary_key: str) -> dict[str, dict[str, float]]:
    methods = ("hybrid", "rl_only", "topsis_only")
    expected_profiles = {"budget", "quality_seeker", "explorer", "loyal", "balanced"}
    sums = {profile: {method: 0.0 for method in methods} for profile in expected_profiles}
    counts = {profile: 0 for profile in expected_profiles}
    catalog_count = 0

    with catalogs_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            catalog_count += 1
            catalog = json.loads(line)
            profiles = catalog["arms"][primary_key]["profiles"]
            require({row["profile_name"] for row in profiles} == expected_profiles, "primary profile set")
            for row in profiles:
                profile = row["profile_name"]
                counts[profile] += 1
                for method in methods:
                    sums[profile][method] += float(row["final_metrics"][method]["f1_at_7"])

    require(catalog_count == 50, "primary profile catalog count")
    require(all(count == 50 for count in counts.values()), "primary profile cell counts")
    means = {
        profile: {method: sums[profile][method] / counts[profile] for method in methods}
        for profile in sorted(expected_profiles)
    }
    expected_means = {
        "balanced": {"hybrid": 0.45428571428571424, "rl_only": 0.12857142857142864, "topsis_only": 0.5800000000000002},
        "budget": {"hybrid": 0.322857142857143, "rl_only": 0.21428571428571427, "topsis_only": 0.19428571428571428},
        "explorer": {"hybrid": 0.3171428571428573, "rl_only": 0.08, "topsis_only": 0.3485714285714286},
        "loyal": {"hybrid": 0.3171428571428573, "rl_only": 0.13714285714285715, "topsis_only": 0.1628571428571429},
        "quality_seeker": {"hybrid": 0.24285714285714297, "rl_only": 0.08571428571428569, "topsis_only": 0.011428571428571429},
    }
    for profile, profile_means in expected_means.items():
        for method, expected in profile_means.items():
            require(math.isclose(means[profile][method], expected, abs_tol=1e-15), f"primary profile mean: {profile}/{method}")
        require(means[profile]["hybrid"] > means[profile]["rl_only"], f"Hybrid-RL profile boundary: {profile}")
    require(means["explorer"]["topsis_only"] > means["explorer"]["hybrid"], "Explorer TOPSIS-Hybrid boundary")
    require(means["balanced"]["topsis_only"] > means["balanced"]["hybrid"], "Balanced TOPSIS-Hybrid boundary")
    for profile in ("budget", "loyal", "quality_seeker"):
        require(means[profile]["hybrid"] > means[profile]["topsis_only"], f"Hybrid-TOPSIS profile boundary: {profile}")
    return means


def verify_semantic_gates() -> dict[str, object]:
    campaign_root = "evidence/campaigns"
    bridge_base = f"{campaign_root}/2026-07-12_codex_local_same_target_bridge_r01_producerenv"
    overlay_base = f"{campaign_root}/2026-07-13_codex_static_verifier_topsis_gatefix"
    drift_base = f"{campaign_root}/2026-07-12_codex_local_same_target_drift_r01_producerenv"
    xai_base = f"{campaign_root}/2026-07-13_codex_local_exact_xai_r1_producerenv"
    latency_base = f"{campaign_root}/2026-07-13_codex_local_latency_r1_producerenv"
    extraction_base = f"{campaign_root}/2026-07-13_codex_verified_payload_extraction_r1"

    bridge_status = load_json(f"{bridge_base}/outputs/canonical_main/status.json")
    require(bridge_status["status"] == "completed_unverified", "bridge terminal status")
    require(bridge_status["runs_completed"] == bridge_status["runs_total"] == 50, "bridge run count")
    require(bridge_status["trajectories_completed"] == bridge_status["trajectories_total"] == 5000, "bridge trajectory count")
    bridge_terminal = ROOT / bridge_base / "outputs/canonical_main/main_results.json"
    require(sha256(bridge_terminal) == bridge_status["terminal_sha256"], "bridge terminal hash")

    original_failure_path = ROOT / bridge_base / "outputs/canonical_main/verification_status.json"
    original_failure = load_json(str(original_failure_path.relative_to(ROOT)).replace("\\", "/"))
    require(original_failure["status"] == "failed", "original verifier history not preserved")
    require(original_failure["catalogs_completed"] == 29, "original verifier failure checkpoint")

    overlay_manifest = load_json(f"{overlay_base}/RUN_MANIFEST.json")
    require(sha256(original_failure_path) == overlay_manifest["source_campaign"]["canonical_failure_status_sha256"], "overlay failure binding")
    overlay_full = load_json(f"{overlay_base}/outputs/canonical_overlay/FULL_VERIFICATION.json")
    require(overlay_full["status"] == "completed_verified" and overlay_full["verdict"] == "PASS", "static overlay verification")
    require(overlay_full["source_campaign_id"] == bridge_status["campaign_id"], "overlay source campaign")
    require(overlay_full["counts"]["catalogs"] == 50, "overlay catalog count")
    require(overlay_full["counts"]["catalog_profile_arm_cells"] == 5000, "overlay catalog-profile-arm count")
    require(overlay_full["counts"]["full_stochastic_replay_cells"] == 5000, "overlay stochastic replay count")
    topsis_evidence = overlay_full["topsis_regression_evidence"]
    require(math.isclose(topsis_evidence["score_abs_tolerance"], 2e-15, abs_tol=0.0), "overlay TOPSIS score tolerance")
    require(topsis_evidence["manual_parser_top7_exact"] == 50, "overlay exact top-7 count")
    require(topsis_evidence["manual_parser_full_400_order_exact"] == 50, "overlay exact full-order count")
    require(math.isclose(topsis_evidence["manual_parser_global_score_max_abs"], 1.1102230246251565e-15, abs_tol=0.0), "overlay TOPSIS max difference")

    drift_full = load_json(f"{drift_base}/outputs/canonical/FULL_VERIFICATION.json")
    require(drift_full["status"] == "PASS", "drift verification")
    xai_full = load_json(f"{xai_base}/outputs/canonical/FULL_VERIFICATION.json")
    require(xai_full["status"] == "completed_verified" and xai_full["verdict"] == "PASS", "XAI verification")
    latency_full = load_json(f"{latency_base}/outputs/canonical/FULL_VERIFICATION.json")
    require(latency_full["status"] == "completed_verified" and latency_full["verdict"] == "PASS", "latency verification")
    extraction_full = load_json(f"{extraction_base}/outputs/canonical/FULL_EXTRACTION_VERIFICATION.json")
    require(extraction_full["status"] == "completed_verified" and extraction_full["verdict"] == "PASS", "extraction verification")

    payload_relative = f"{extraction_base}/outputs/canonical/VERIFIED_SCIENTIFIC_PAYLOAD.json"
    payload_path = ROOT / payload_relative
    # The canonical FULL report binds the original payload. The public copy
    # replaces only machine-path strings. The privacy manifest binds both
    # hashes and records that no scientific numeric leaf changed.
    redaction = load_json("evidence/privacy/REDACTION_MANIFEST.json")
    require(redaction["status"] == "PASS" and redaction["scientific_numeric_redactions"] == 0, "privacy redaction gate")
    records = {record["path"]: record for record in redaction["records"]}
    payload_original_relative = payload_relative.replace("evidence/campaigns/", "experiments/", 1)
    payload_record = records[payload_original_relative]
    require(payload_record["canonical_original_sha256"] == extraction_full["payload_sha256"], "canonical payload binding")
    require(payload_record["privacy_safe_sha256"] == sha256(payload_path), "privacy-safe payload binding")
    require(payload_record["scientific_content_change"] is False, "payload scientific redaction")
    payload = json.loads(payload_path.read_text(encoding="utf-8-sig"))

    primary_key = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
    primary = payload["static"]["analysis"]["summaries"][primary_key]
    require(math.isclose(primary["methods"]["hybrid"]["mean"], 0.3308571428571428, abs_tol=1e-15), "primary Hybrid F1")
    require(math.isclose(primary["methods"]["rl_only"]["mean"], 0.12914285714285714, abs_tol=1e-15), "primary RL F1")
    require(math.isclose(primary["methods"]["topsis_only"]["mean"], 0.2594285714285714, abs_tol=1e-15), "primary TOPSIS F1")
    require(primary["hybrid_minus_rl"]["wins"] == 50 and primary["hybrid_minus_rl"]["losses"] == 0, "primary Hybrid-RL wins")
    require(primary["hybrid_minus_topsis"]["wins"] == 34 and primary["hybrid_minus_topsis"]["losses"] == 12 and primary["hybrid_minus_topsis"]["ties"] == 4, "primary Hybrid-TOPSIS wins")
    require(math.isclose(primary["hybrid_minus_rl"]["mean"], 0.20171428571428568, abs_tol=1e-15), "primary Hybrid-RL difference")
    require(math.isclose(primary["hybrid_minus_rl"]["bootstrap_ci95_lo"], 0.18171428571428566, abs_tol=1e-15), "primary Hybrid-RL CI low")
    require(math.isclose(primary["hybrid_minus_rl"]["bootstrap_ci95_hi"], 0.22114285714285714, abs_tol=1e-15), "primary Hybrid-RL CI high")
    require(math.isclose(primary["hybrid_minus_topsis"]["mean"], 0.07142857142857141, abs_tol=1e-15), "primary Hybrid-TOPSIS difference")
    require(math.isclose(primary["hybrid_minus_topsis"]["bootstrap_ci95_lo"], 0.044571428571428574, abs_tol=1e-15), "primary Hybrid-TOPSIS CI low")
    require(math.isclose(primary["hybrid_minus_topsis"]["bootstrap_ci95_hi"], 0.0982857142857143, abs_tol=1e-15), "primary Hybrid-TOPSIS CI high")

    verify_primary_profile_boundary(
        ROOT / bridge_base / "outputs/canonical_main/main_catalogs.jsonl",
        primary_key,
    )

    exact_key = "candidate=oracle_gt_hidden30__bonus=0.20__reward=implemented_r0"
    exact_r0 = payload["static"]["analysis"]["summaries"][exact_key]
    require(math.isclose(exact_r0["methods"]["hybrid"]["mean"], 0.9005714285714284, abs_tol=1e-15), "exact-r0 Hybrid F1")
    require(math.isclose(exact_r0["methods"]["rl_only"]["mean"], 0.5742857142857143, abs_tol=1e-15), "exact-r0 RL F1")

    reward_model = payload["drift"]["analysis"]["primary_reward_model"]
    sudden_gap = payload["drift"]["analysis"]["sudden"][reward_model]["paired_hybrid_minus_rl"]["final_f1"]["mean"]
    gradual_gap = payload["drift"]["analysis"]["gradual"][reward_model]["paired_hybrid_minus_rl"]["final_f1"]["mean"]
    require(math.isclose(sudden_gap, 0.252, abs_tol=1e-15), "sudden drift gap")
    require(math.isclose(gradual_gap, 0.2295238095238095, abs_tol=1e-15), "gradual drift gap")

    xai_errors = payload["exact_xai"]["diagnostic_max_abs_error"]
    require(max(float(value) for value in xai_errors.values()) <= 1e-12, "XAI exact reconstruction")
    latency = payload["latency"]["raw_all_samples"]
    require(latency["n"] == 300000, "latency sample count")
    require(latency["median_ns"] == 53500.0 and latency["p95_ns"] == 117600.0 and latency["p99_ns"] == 150200.0, "latency quantiles")

    mccauley = load_json("evidence/legacy_external_boundary/results/mccauley_home_real_results.json")
    epoch30 = {method: float(mccauley["summary"][method]["30"]["mean"]) for method in ("hybrid", "rl_only", "topsis_only", "popularity", "random")}
    expected_mccauley = {
        "hybrid": 0.06579443658651579,
        "rl_only": 0.05594648750589344,
        "topsis_only": 0.06424446016030173,
        "popularity": 0.13009193776520508,
        "random": 0.039085337105139076,
    }
    require(all(math.isclose(epoch30[method], value, abs_tol=1e-15) for method, value in expected_mccauley.items()), "McAuley epoch-30 means")
    require(epoch30["popularity"] > epoch30["hybrid"] > epoch30["topsis_only"], "McAuley negative boundary")

    return {
        "bridge_compute": "completed_50_of_50",
        "original_verifier_record": "preserved_failed_closed",
        "static_overlay": "completed_verified/PASS",
        "drift": "PASS",
        "exact_xai": "completed_verified/PASS",
        "latency": "completed_verified/PASS",
        "payload_extraction": "completed_verified/PASS",
        "profile_conditionality": "verified",
        "mccauley_boundary": "popularity_gt_hybrid_gt_topsis",
    }


def main() -> int:
    try:
        file_count, total_bytes = verify_inventory()
        verify_privacy_boundary()
        gates = verify_semantic_gates()
        report = {
            "schema_version": "hre.public_evidence_verification.v1",
            "verdict": "PASS",
            "file_count": file_count,
            "total_bytes": total_bytes,
            "gates": gates,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (EvidenceError, KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
