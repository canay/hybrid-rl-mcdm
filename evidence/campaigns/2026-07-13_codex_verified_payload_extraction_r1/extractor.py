from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CAMPAIGN_ID = "2026-07-13_codex_verified_payload_extraction_r1"
OPERATION_ID = "HRE_R1_VERIFIED_SCIENCE_EXTRACTION_20260713_CODEX_25"
CREATED_AT = "2026-07-13T05:14:00+03:00"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "outputs" / "canonical"

STATIC = ROOT / "experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv"
OVERLAY = ROOT / "experiments/2026-07-13_codex_static_verifier_topsis_gatefix"
DRIFT = ROOT / "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv"
LATENCY = ROOT / "experiments/2026-07-13_codex_local_latency_r1_producerenv"
XAI = ROOT / "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv"

EXPECTED_CRITICAL = {
    "experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv/RUN_MANIFEST.json": "0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222",
    "experiments/2026-07-13_codex_static_verifier_topsis_gatefix/RUN_MANIFEST.json": "ab29bb4782daf44c856f79ef8ad83559d4b59b637409476a43ed02dac8672ea7",
    "experiments/2026-07-13_codex_static_verifier_topsis_gatefix/outputs/canonical_overlay/FULL_VERIFICATION.json": "a3112da73a28e9c68f3148b0a8668cc834472c7d5c765870ad1fed25e09fcd97",
    "experiments/2026-07-13_codex_static_verifier_topsis_gatefix/outputs/canonical_overlay/verification_status.json": "2513c8922a38e5f1a413d081b607225a55a620d7ad1cf540db208c48373b811f",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/RUN_MANIFEST.json": "21ef380e9ab68ca259d04abbf0ebe5c121e6280dbcbe50cb366211726c227800",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/FULL_VERIFICATION.json": "acb62f386a362c0292e275e7f2ca657f3760016a58860aab206bf760eb7934b7",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/STATUS.json": "c3168815e4b22dcdc116aa32f481770051506b9e8e9b05617a3a603bbd7e5fba",
    "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical/VERIFICATION_STATUS.json": "7b061214ad78f59319d6ad453c573dfca163787f9f0d9d66bd6ad109b3abc126",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/RUN_MANIFEST.json": "6882904f426795622f833256d143db111903913abb51e0e1cc539a3bdb89be68",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/FULL_VERIFICATION.json": "93534ba86f117dc1884b02b6c0e46cea936954a2ff1ba8935dcf85c098a0a5c3",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/status.json": "e8b080b2bad9df3e6141166b9066a47af16ba43a41a054ff63d9e1be621d1432",
    "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical/verification_status.json": "05c040c1c91c1cc37049ed75bac265373858d03734ccf22dd44180f935cfc1fa",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/RUN_MANIFEST.json": "ff691ebcb1a85d35057a9baba72a360b0034c72e7139e41009b98d0575cd27e6",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/FULL_VERIFICATION.json": "a0c53db03e36b4a76b9870f1d910184d134c6ce343ef64ce8a4a3ec413d3e7bd",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/status.json": "78a5b8386d47c7354f6ca121d3c30d019cd663248c91adab33ce8250b7fb1020",
    "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical/verification_status.json": "9b051087de8ce2b72fa8e0af5c3e233b463ff03447c1a33ea321808db8004501",
}

ANCHORS = {
    "exact_r0_reproduction": "candidate=oracle_gt_hidden30__bonus=0.20__reward=implemented_r0",
    "internal_component_repair": "candidate=oracle_gt_hidden30__bonus=0.20__reward=component_continuous_fix",
    "primary_corrected": "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix",
    "literal_fix_sensitivity": "candidate=full_catalog__bonus=0.00__reward=inclusive_range_fix",
    "historical_funnel_sensitivity": "candidate=full_catalog__bonus=0.00__reward=historical_funnel_coefficients_on_may_h",
}


class ExtractionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value: str) -> None:
    raise ExtractionError(f"Non-finite JSON constant: {value}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExtractionError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        require(bool(line.strip()), f"Blank JSONL line {index}: {path}")
        value = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_object)
        require(isinstance(value, dict), f"JSONL record is not an object: {path}:{index + 1}")
        records.append(value)
    return records


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_run_manifest() -> dict[str, Any]:
    path = HERE / "RUN_MANIFEST.json"
    manifest = load_json(path)
    require(manifest.get("schema_version") == "hre.verified_science_extraction.run_manifest.v1", "Bad extraction manifest schema")
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "Bad extraction campaign ID")
    require(manifest.get("status") == "LOCKED_BEFORE_EXTRACTION", "Extraction manifest is not locked")
    for section, base in (("locked_local_files", HERE), ("upstream_files", ROOT)):
        for row in manifest[section]:
            target = base / row["path"]
            require(target.is_file(), f"Missing locked file: {target}")
            require(target.stat().st_size == row["bytes"], f"Locked size mismatch: {target}")
            require(sha256(target) == row["sha256"], f"Locked hash mismatch: {target}")
    for relative, expected in EXPECTED_CRITICAL.items():
        require(sha256(ROOT / relative) == expected, f"Critical hash mismatch: {relative}")
    return manifest


def require_gates(rows: list[dict[str, Any]], expected_count: int, label: str) -> None:
    require(len(rows) == expected_count, f"{label} gate count mismatch")
    require(all(row.get("status") == "PASS" for row in rows), f"{label} contains a non-PASS gate")


def verify_sources() -> dict[str, Any]:
    # Static source is deliberately authorized by the separate overlay.
    static_root = STATIC / "outputs/canonical_main"
    overlay_root = OVERLAY / "outputs/canonical_overlay"
    overlay_full = load_json(overlay_root / "FULL_VERIFICATION.json")
    overlay_status = load_json(overlay_root / "verification_status.json")
    static_status = load_json(static_root / "status.json")
    require(overlay_full.get("status") == "completed_verified" and overlay_full.get("verdict") == "PASS", "Static overlay is not terminal PASS")
    require_gates(overlay_full.get("gates", []), 10, "static overlay")
    require(overlay_status.get("status") == "completed_verified", "Static overlay status is not completed_verified")
    require(overlay_status.get("full_verification_sha256") == sha256(overlay_root / "FULL_VERIFICATION.json"), "Static overlay FULL hash link mismatch")
    require(static_status.get("status") == "completed_unverified", "Static runner state mismatch")
    for filename, expected in overlay_full["output_hashes"].items():
        require(sha256(static_root / filename) == expected, f"Static output hash mismatch: {filename}")
    require(static_status.get("terminal_sha256") == sha256(static_root / "main_results.json"), "Static terminal link mismatch")

    drift_root = DRIFT / "outputs/canonical"
    drift_full = load_json(drift_root / "FULL_VERIFICATION.json")
    drift_status = load_json(drift_root / "STATUS.json")
    drift_verify = load_json(drift_root / "VERIFICATION_STATUS.json")
    drift_terminal = load_json(drift_root / "TERMINAL.json")
    require(drift_full.get("status") == "PASS", "Drift FULL is not PASS")
    require(drift_status.get("status") == "completed_verified" and drift_verify.get("status") == "completed_verified", "Drift terminal status mismatch")
    require(drift_terminal.get("status") == "completed_unverified", "Drift runner terminal mismatch")
    require(drift_status.get("full_verification_sha256") == sha256(drift_root / "FULL_VERIFICATION.json"), "Drift FULL hash link mismatch")
    require(drift_status.get("terminal_sha256") == sha256(drift_root / "TERMINAL.json"), "Drift terminal hash link mismatch")
    require(drift_status.get("sealed_records_sha256") == sha256(drift_root / "sealed_records.jsonl"), "Drift sealed hash link mismatch")
    require(drift_verify.get("full_verification_sha256") == sha256(drift_root / "FULL_VERIFICATION.json"), "Drift verifier FULL link mismatch")
    require(drift_full["hash_chain"]["terminal_sha256"] == sha256(drift_root / "TERMINAL.json"), "Drift FULL terminal binding mismatch")
    require(drift_full["hash_chain"]["sealed_records_sha256"] == sha256(drift_root / "sealed_records.jsonl"), "Drift FULL sealed binding mismatch")

    latency_root = LATENCY / "outputs/canonical"
    latency_full = load_json(latency_root / "FULL_VERIFICATION.json")
    latency_status = load_json(latency_root / "status.json")
    latency_verify = load_json(latency_root / "verification_status.json")
    require(latency_full.get("status") == "completed_verified" and latency_full.get("verdict") == "PASS", "Latency FULL is not PASS")
    require(latency_status.get("status") == "completed_verified" and latency_status.get("verdict") == "PASS", "Latency status is not PASS")
    require(latency_verify.get("status") == "completed_verified" and latency_verify.get("verdict") == "PASS", "Latency verifier is not PASS")
    require(latency_status.get("terminal_sha256") == sha256(latency_root / "FULL_VERIFICATION.json"), "Latency status FULL link mismatch")
    require(latency_verify.get("terminal_sha256") == sha256(latency_root / "FULL_VERIFICATION.json"), "Latency verifier FULL link mismatch")
    for filename, expected in latency_full["output_hashes"].items():
        require(sha256(latency_root / filename) == expected, f"Latency output hash mismatch: {filename}")

    xai_root = XAI / "outputs/canonical"
    xai_full = load_json(xai_root / "FULL_VERIFICATION.json")
    xai_status = load_json(xai_root / "status.json")
    xai_verify = load_json(xai_root / "verification_status.json")
    require(xai_full.get("status") == "completed_verified" and xai_full.get("verdict") == "PASS", "XAI FULL is not PASS")
    require_gates(xai_full.get("gates", []), 9, "XAI")
    require(xai_status.get("status") == "completed_unverified", "XAI runner terminal mismatch")
    require(xai_verify.get("status") == "completed_verified", "XAI verifier is not completed_verified")
    require(xai_verify.get("profile_cells_verified") == 250, "XAI verified cell count mismatch")
    for filename, expected in xai_full["output_hashes"].items():
        require(sha256(xai_root / filename) == expected, f"XAI output hash mismatch: {filename}")
    require(xai_status.get("terminal_sha256") == sha256(xai_root / "xai_results.json"), "XAI terminal link mismatch")

    return {
        "static_overlay_full": overlay_full,
        "drift_full": drift_full,
        "latency_full": latency_full,
        "xai_full": xai_full,
    }


def ci_values(row: Mapping[str, Any]) -> tuple[Any, Any]:
    if "bootstrap_ci95" in row:
        return row["bootstrap_ci95"][0], row["bootstrap_ci95"][1]
    return row.get("bootstrap_ci95_lo", ""), row.get("bootstrap_ci95_hi", "")


def bootstrap_summary(values: Iterable[float], label: str) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    require(arr.ndim == 1 and arr.size > 0 and np.all(np.isfinite(arr)), f"Invalid bootstrap vector: {label}")
    seed = int.from_bytes(hashlib.sha256(f"verified_extraction_v1|{label}".encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(20000, arr.size))].mean(axis=1)
    return {
        "mean": float(arr.mean()), "sample_sd": sample_sd(arr),
        "ci95_lo": float(np.quantile(means, 0.025)), "ci95_hi": float(np.quantile(means, 0.975)),
        "n_catalog_resamples": int(arr.size), "bootstrap_reps": 20000, "bootstrap_seed": seed,
    }


def static_outputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = STATIC / "outputs/canonical_main"
    terminal = load_json(root / "main_results.json")
    require(terminal.get("schema_version") == "same_target_bridge.terminal.v1", "Static terminal schema mismatch")
    analysis = terminal.get("analysis", {})
    summaries = analysis.get("summaries", {})
    require(len(summaries) == 20 and analysis.get("arm_count") == 20, "Static arm count mismatch")
    raw = load_jsonl(root / "main_catalogs.jsonl")
    require(len(raw) == 50, "Static raw catalog count mismatch")
    for index, record in enumerate(raw):
        require(record.get("run_index") == index, "Static run order mismatch")
        require(len(record.get("arms", {})) == 20, "Static raw arm count mismatch")
        require(all(len(arm.get("profiles", [])) == 5 for arm in record["arms"].values()), "Static profile count mismatch")

    arm_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    for arm_id, summary in summaries.items():
        arm = summary["arm"]
        for metric, metric_row in (("f1_at_7", summary), ("ndcg_at_7", summary["ndcg_at_7"])):
            for method, stats in metric_row["methods"].items():
                lo, hi = ci_values(stats)
                row = {
                    "metric": metric,
                    "arm_id": arm_id,
                    "candidate": arm["candidate"],
                    "gt_bonus": arm["gt_bonus"],
                    "reward_model": arm["reward_model"],
                    "role": arm["role"],
                    "method": method,
                    "mean": stats["mean"],
                    "sample_sd": stats["sample_sd"],
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_catalogs": stats["n_catalogs"],
                }
                arm_rows.append(row)
            anchor_name = next((name for name, value in ANCHORS.items() if value == arm_id), None)
            if anchor_name:
                for name in ("hybrid", "rl_only", "topsis_only"):
                    stats = metric_row["methods"][name]
                    lo, hi = ci_values(stats)
                    anchor_rows.append({
                        "anchor": anchor_name, "metric": metric, "row_type": "method", "name": name,
                        "arm_id": arm_id, "mean": stats["mean"], "sample_sd": stats["sample_sd"],
                        "ci95_lo": lo, "ci95_hi": hi, "n_catalogs": stats["n_catalogs"],
                        "wins": "", "ties": "", "losses": "", "paired_t_stat": "",
                        "paired_t_p_two_sided": "", "wilcoxon_stat": "", "wilcoxon_p_two_sided": "",
                        "cohen_dz": "", "cohen_dz_defined": "",
                    })
                for field, name in (("hybrid_minus_rl", "hybrid_minus_rl"), ("hybrid_minus_topsis", "hybrid_minus_topsis")):
                    stats = metric_row[field]
                    lo, hi = ci_values(stats)
                    anchor_rows.append({
                        "anchor": anchor_name, "metric": metric, "row_type": "paired_difference", "name": name,
                        "arm_id": arm_id, "mean": stats["mean"], "sample_sd": stats["sample_sd"],
                        "ci95_lo": lo, "ci95_hi": hi, "n_catalogs": stats["n_catalogs"],
                        "wins": stats["wins"], "ties": stats["ties"], "losses": stats["losses"],
                        "paired_t_stat": stats["paired_t_stat"], "paired_t_p_two_sided": stats["paired_t_p_two_sided"],
                        "wilcoxon_stat": stats["wilcoxon_stat"], "wilcoxon_p_two_sided": stats["wilcoxon_p_two_sided"],
                        "cohen_dz": stats["cohen_dz"], "cohen_dz_defined": stats["cohen_dz_defined"],
                    })

    contrast_rows: list[dict[str, Any]] = []
    for family in ("factorial_contrasts", "sensitivity_contrasts"):
        for metric, contrasts in analysis[family].items():
            for contrast_id, contrast in contrasts.items():
                specification = contrast["specification"]
                for method in ("hybrid", "rl_only"):
                    stats = contrast[method]
                    lo, hi = ci_values(stats)
                    contrast_rows.append({
                        "family": family, "metric": metric, "contrast_id": contrast_id, "method": method,
                        "factor": specification.get("factor", "reward_sensitivity"),
                        "direction": specification.get("direction", contrast_id),
                        "arm_a": specification["arm_a"], "arm_b": specification["arm_b"],
                        "mean": stats["mean"], "sample_sd": stats["sample_sd"], "ci95_lo": lo, "ci95_hi": hi,
                        "n_catalogs": stats["n_catalogs"], "wins": stats["wins"], "ties": stats["ties"], "losses": stats["losses"],
                        "paired_t_stat": stats["paired_t_stat"], "paired_t_p_two_sided": stats["paired_t_p_two_sided"],
                        "wilcoxon_stat": stats["wilcoxon_stat"], "wilcoxon_p_two_sided": stats["wilcoxon_p_two_sided"],
                        "cohen_dz": stats["cohen_dz"], "cohen_dz_defined": stats["cohen_dz_defined"],
                    })

    primary = summaries[ANCHORS["primary_corrected"]]
    primary_gate = {}
    for metric, metric_row in (("f1_at_7", primary), ("ndcg_at_7", primary["ndcg_at_7"])):
        comparisons = {name: metric_row[name] for name in ("hybrid_minus_rl", "hybrid_minus_topsis")}
        primary_gate[metric] = {
            "hybrid_minus_rl_mean_positive": comparisons["hybrid_minus_rl"]["mean"] > 0,
            "hybrid_minus_rl_ci_lower_positive": comparisons["hybrid_minus_rl"]["bootstrap_ci95_lo"] > 0,
            "hybrid_minus_topsis_mean_positive": comparisons["hybrid_minus_topsis"]["mean"] > 0,
            "hybrid_minus_topsis_ci_lower_positive": comparisons["hybrid_minus_topsis"]["bootstrap_ci95_lo"] > 0,
        }
        primary_gate[metric]["strong_complementarity_gate"] = all(primary_gate[metric].values())

    diagnostic_rows: list[dict[str, Any]] = []
    profile_order = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]
    for anchor_name in ("primary_corrected", "exact_r0_reproduction"):
        arm_id = ANCHORS[anchor_name]
        checkpoints = sorted(raw[0]["arms"][arm_id]["profiles"][0]["checkpoint_f1"], key=int)
        # All-method overall checkpoint curves: profiles averaged within each run.
        for checkpoint in checkpoints:
            for method in ("hybrid", "rl_only", "topsis_only", "popularity", "random"):
                vector = [float(np.mean([profile["checkpoint_f1"][checkpoint][method] for profile in record["arms"][arm_id]["profiles"]])) for record in raw]
                stats = bootstrap_summary(vector, f"{anchor_name}|checkpoint_overall|{checkpoint}|{method}")
                diagnostic_rows.append({"anchor": anchor_name, "row_type": "checkpoint_overall", "profile": "all_profiles_mean_within_run", "metric": "f1_at_7", "checkpoint": checkpoint, "method": method, **stats})
        # Fixed-profile Hybrid/RL checkpoint curves. These are descriptive profile strata.
        for profile_index, profile_name in enumerate(profile_order):
            for checkpoint in checkpoints:
                for method in ("hybrid", "rl_only"):
                    vector = [record["arms"][arm_id]["profiles"][profile_index]["checkpoint_f1"][checkpoint][method] for record in raw]
                    stats = bootstrap_summary(vector, f"{anchor_name}|checkpoint_profile|{profile_name}|{checkpoint}|{method}")
                    diagnostic_rows.append({"anchor": anchor_name, "row_type": "checkpoint_profile_descriptive", "profile": profile_name, "metric": "f1_at_7", "checkpoint": checkpoint, "method": method, **stats})
            # Per-profile final F1/NDCG for every method.
            for metric in ("f1_at_7", "ndcg_at_7"):
                for method in ("hybrid", "rl_only", "topsis_only", "popularity", "random"):
                    vector = [record["arms"][arm_id]["profiles"][profile_index]["final_metrics"][method][metric] for record in raw]
                    stats = bootstrap_summary(vector, f"{anchor_name}|final_profile|{profile_name}|{metric}|{method}")
                    diagnostic_rows.append({"anchor": anchor_name, "row_type": "final_profile_descriptive", "profile": profile_name, "metric": metric, "checkpoint": 30000, "method": method, **stats})

    return {"analysis": analysis, "primary_claim_gate": primary_gate}, arm_rows, anchor_rows, contrast_rows, diagnostic_rows


def sample_sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def drift_outputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = DRIFT / "outputs/canonical"
    terminal = load_json(root / "TERMINAL.json")
    require(terminal.get("schema_version") == "same_target_drift.terminal.v1", "Drift terminal schema mismatch")
    records = load_jsonl(root / "sealed_records.jsonl")
    require(len(records) == 50, "Drift record count mismatch")
    analysis = terminal["analysis"]
    require(analysis.get("primary_reward_model") == "component_continuous_fix", "Drift primary model mismatch")
    rows: list[dict[str, Any]] = []
    for scenario in ("sudden", "gradual"):
        require(set(analysis[scenario]) == {"component_continuous_fix", "inclusive_range_fix", "historical_funnel_coefficients_on_may_h"}, f"Drift {scenario} model set mismatch")
        for reward_model, condition in analysis[scenario].items():
            for name in ("hybrid", "rl_only", "paired_hybrid_minus_rl"):
                for metric in ("final_f1", "checkpoint_normalized_post_change_auc"):
                    stats = condition[name][metric]
                    lo, hi = ci_values(stats)
                    rows.append({
                        "condition_class": "future_blind", "scenario": scenario, "reward_model": reward_model,
                        "row_type": "paired_difference" if name == "paired_hybrid_minus_rl" else "method",
                        "name": name, "metric": metric, "checkpoint": "", "mean": stats["mean"],
                        "sample_sd": stats["sample_sd"], "ci95_lo": lo, "ci95_hi": hi,
                        "n_catalog_resamples": stats["n_catalog_resamples"], "wins": stats.get("wins", ""),
                        "ties": stats.get("ties", ""), "losses": stats.get("losses", ""),
                        "raw_catalog_resample_vector_json": json.dumps(stats.get("raw_catalog_resample_vector", []), separators=(",", ":")),
                    })

    legacy_summary: dict[str, Any] = {}
    for scenario, run_count in (("sudden", 50), ("gradual", 30)):
        first = records[0][scenario]["legacy_exact"]
        checkpoints = sorted(first[0]["checkpoints"], key=int)
        methods = sorted(first[0]["checkpoints"][checkpoints[0]]["f1"])
        legacy_summary[scenario] = {}
        for checkpoint in checkpoints:
            legacy_summary[scenario][checkpoint] = {}
            for method in methods:
                vector = []
                for record in records[:run_count]:
                    profiles = record[scenario]["legacy_exact"]
                    vector.append(float(np.mean([profile["checkpoints"][checkpoint]["f1"][method] for profile in profiles])))
                arr = np.asarray(vector, dtype=float)
                summary = {"mean": float(arr.mean()), "sample_sd": sample_sd(arr), "n_catalog_resamples": int(arr.size)}
                legacy_summary[scenario][checkpoint][method] = summary
                rows.append({
                    "condition_class": "legacy_exact_reproduction", "scenario": scenario, "reward_model": "implemented_r0",
                    "row_type": "method", "name": method, "metric": "checkpoint_f1", "checkpoint": checkpoint,
                    "mean": summary["mean"], "sample_sd": summary["sample_sd"], "ci95_lo": "", "ci95_hi": "",
                    "n_catalog_resamples": summary["n_catalog_resamples"], "wins": "", "ties": "", "losses": "",
                    "raw_catalog_resample_vector_json": json.dumps(vector, separators=(",", ":")),
                })
    return {"analysis": analysis, "legacy_exact_reproduction": legacy_summary}, rows


def latency_output() -> dict[str, Any]:
    root = LATENCY / "outputs/canonical"
    result = load_json(root / "latency_results.json")
    require(result.get("schema_version") == "hre.latency_result.v1", "Latency result schema mismatch")
    require(result.get("status") == "completed_unverified", "Latency runner state mismatch")
    require(result["accuracy"]["all_exact"] is True, "Latency accuracy gate failed")
    require(len(result["accuracy"]["before"]) == 250 and len(result["accuracy"]["after"]) == 250, "Latency pair count mismatch")
    require(result["retention_gate"]["status"] == "PASS_RETAIN_CACHED_PATH_CLAIM", "Latency retention gate failed")
    return {
        "schema_version": "hre.verified_latency_payload.v1",
        "claim_boundary": result["claim_boundary"],
        "config": result["config"],
        "pair_schedule": result["pair_schedule"],
        "raw_all_samples": result["analysis"]["raw_all_samples"],
        "passes_all_samples": [{"pass_index": row["pass_index"], "all_samples": row["all_samples"]} for row in result["analysis"]["passes"]],
        "stable_all_samples_secondary": result["analysis"]["stable_all_samples"],
        "threshold_diagnostics": result["analysis"]["threshold_diagnostics"],
        "retention_gate": result["retention_gate"],
        "accuracy": {"all_exact": result["accuracy"]["all_exact"], "before_count": len(result["accuracy"]["before"]), "after_count": len(result["accuracy"]["after"])},
        "environment": result["environment"],
        "timer_overhead": result["timer_overhead"],
        "raw_artifact": result["raw_artifact"],
    }


def distribution(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    require(arr.ndim == 1 and arr.size > 0 and np.all(np.isfinite(arr)), "Invalid summary vector")
    return {
        "n_catalog_resamples": int(arr.size),
        "mean": float(arr.mean()),
        "sample_sd": sample_sd(arr),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(np.percentile(arr, 5, method="linear")),
        "p95": float(np.percentile(arr, 95, method="linear")),
    }


def xai_outputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = XAI / "outputs/canonical"
    inputs = load_jsonl(root / "xai_inputs.jsonl")
    attrs = load_jsonl(root / "xai_attributions.jsonl")
    require(len(inputs) == len(attrs) == 50, "XAI record count mismatch")
    forbidden = ("gt", "ground_truth", "target", "label", "metric", "f1", "ndcg", "relevance")

    def inspect_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                low = key.lower()
                require(not any(token in low for token in forbidden), f"Forbidden XAI input key: {key}")
                inspect_keys(child)
        elif isinstance(value, list):
            for child in value:
                inspect_keys(child)

    run_rows: list[dict[str, Any]] = []
    expected_profiles = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]
    criteria: list[str] | None = None
    for index, (input_row, row) in enumerate(zip(inputs, attrs)):
        inspect_keys(input_row)
        require(input_row["run_index"] == row["run_index"] == index, "XAI run order mismatch")
        require(input_row["allowlisted_payload_sha256"] == row["allowlisted_payload_sha256"], "XAI allowlist hash mismatch")
        profile_names = [profile["profile_name"] for profile in row["profiles"]]
        require(profile_names == expected_profiles, "XAI profile order mismatch")
        current_criteria = row["topsis"]["criteria"]
        if criteria is None:
            criteria = current_criteria
        require(current_criteria == criteria == ["price_pct", "quality_pct", "popularity_pct", "rating_pct"], "XAI criteria order mismatch")
        shapley = np.asarray(row["topsis"]["shapley_values_normalized"], dtype=float)
        require(shapley.shape == (400, 4), "XAI Shapley shape mismatch")

        selected_shapley = []
        selected_c_t = []
        selected_c_q = []
        selected_hybrid = []
        selected_components = {name: [] for name in ("base", "engage", "convert")}
        profile_details = []
        for profile in row["profiles"]:
            indices = np.asarray(profile["hybrid_top7_rank"], dtype=int)
            require(indices.shape == (7,) and np.all((indices >= 0) & (indices < 400)), "XAI top-7 shape/range mismatch")
            selected_shapley.append(shapley[indices])
            selected_c_t.extend(np.asarray(profile["c_t"], dtype=float)[indices].tolist())
            selected_c_q.extend(np.asarray(profile["c_q"], dtype=float)[indices].tolist())
            selected_hybrid.extend(np.asarray(profile["hybrid_score"], dtype=float)[indices].tolist())
            component_summary = {}
            for name in selected_components:
                values = np.asarray(profile[f"c_q_{name}"], dtype=float)[indices]
                selected_components[name].extend(values.tolist())
                component_summary[name] = {"signed_mean": float(values.mean()), "mean_abs": float(np.abs(values).mean())}
            profile_details.append({
                "profile_name": profile["profile_name"],
                "top7_rank": indices.tolist(),
                "reward_component_selected_top7": component_summary,
                "fusion_selected_top7": {
                    "c_t_mean": float(np.asarray(profile["c_t"], dtype=float)[indices].mean()),
                    "c_q_mean": float(np.asarray(profile["c_q"], dtype=float)[indices].mean()),
                    "hybrid_score_mean": float(np.asarray(profile["hybrid_score"], dtype=float)[indices].mean()),
                },
                "engage_rate": profile["diagnostics"]["engage_count"] / 30000.0,
                "convert_rate": profile["diagnostics"]["convert_count"] / 30000.0,
            })
        selected_shapley_arr = np.concatenate(selected_shapley, axis=0)
        component_run = {}
        for name, values in selected_components.items():
            arr = np.asarray(values, dtype=float)
            component_run[name] = {"signed_mean": float(arr.mean()), "mean_abs": float(np.abs(arr).mean())}
        run_rows.append({
            "schema_version": "hre.exact_xai_run_level_summary.v1",
            "run_index": index,
            "run_seed": row["run_seed"],
            "dataset_sha256": row["dataset_sha256"],
            "criteria": criteria,
            "topsis_weights": row["topsis"]["weights"],
            "topsis_shapley_all_items": {
                "signed_mean": np.mean(shapley, axis=0).tolist(),
                "mean_abs": np.mean(np.abs(shapley), axis=0).tolist(),
            },
            "topsis_shapley_selected_top7_profile_weighted": {
                "signed_mean": np.mean(selected_shapley_arr, axis=0).tolist(),
                "mean_abs": np.mean(np.abs(selected_shapley_arr), axis=0).tolist(),
            },
            "topsis_shapley_selected_top7_hybrid_score_space": {
                "signed_mean": (0.5 * np.mean(selected_shapley_arr, axis=0)).tolist(),
                "mean_abs": (0.5 * np.mean(np.abs(selected_shapley_arr), axis=0)).tolist(),
            },
            "reward_component_selected_top7_profile_weighted": component_run,
            "fusion_selected_top7_profile_weighted": {
                "c_t_mean": float(np.mean(selected_c_t)), "c_q_mean": float(np.mean(selected_c_q)),
                "hybrid_score_mean": float(np.mean(selected_hybrid)),
            },
            "event_rates_profile_weighted": {
                "engage_rate": float(np.mean([p["engage_rate"] for p in profile_details])),
                "convert_rate": float(np.mean([p["convert_rate"] for p in profile_details])),
            },
            "diagnostic_max_abs_error": {
                "topsis_efficiency": row["topsis"]["diagnostics"]["shapley_efficiency_max_abs_error"],
                "topsis_reconstruction": row["topsis"]["diagnostics"]["normalized_reconstruction_max_abs_error"],
                "q_component": max(p["diagnostics"]["q_component_raw_reconstruction_max_abs_error"] for p in row["profiles"]),
                "q_affine": max(p["diagnostics"]["c_q_affine_reconstruction_max_abs_error"] for p in row["profiles"]),
                "hybrid": max(p["diagnostics"]["hybrid_reconstruction_max_abs_error"] for p in row["profiles"]),
                "source_q_replay": max(p["diagnostics"]["source_q_replay_max_abs_error"] for p in row["profiles"]),
            },
            "profile_details_descriptive_only": profile_details,
        })

    assert criteria is not None
    summary: dict[str, Any] = {
        "schema_version": "hre.exact_xai_summary.v1",
        "aggregation_unit": "catalog-resample; five fixed profiles averaged within run",
        "interpretation_boundary": "Exact fixed-reference TOPSIS decomposition and realized-policy reward decomposition; not causal, counterfactual, or population-preference evidence.",
        "criteria": criteria,
        "catalog_resamples": 50,
        "profile_cells": 250,
        "subsets_predeclared": ["all_400_items", "hybrid_selected_top7_profile_weighted"],
        "topsis_weights": {},
        "topsis_shapley_all_items": {"signed_mean": {}, "mean_abs": {}},
        "topsis_shapley_selected_top7_profile_weighted": {"signed_mean": {}, "mean_abs": {}},
        "topsis_shapley_selected_top7_hybrid_score_space": {"signed_mean": {}, "mean_abs": {}},
        "reward_component_selected_top7_profile_weighted": {},
        "fusion_selected_top7_profile_weighted": {},
        "event_rates_profile_weighted": {},
        "diagnostic_max_abs_error": {},
    }
    for ci, criterion in enumerate(criteria):
        summary["topsis_weights"][criterion] = distribution(row["topsis_weights"][ci] for row in run_rows)
        for section in ("topsis_shapley_all_items", "topsis_shapley_selected_top7_profile_weighted", "topsis_shapley_selected_top7_hybrid_score_space"):
            for statistic in ("signed_mean", "mean_abs"):
                summary[section][statistic][criterion] = distribution(row[section][statistic][ci] for row in run_rows)
    for component in ("base", "engage", "convert"):
        summary["reward_component_selected_top7_profile_weighted"][component] = {
            statistic: distribution(row["reward_component_selected_top7_profile_weighted"][component][statistic] for row in run_rows)
            for statistic in ("signed_mean", "mean_abs")
        }
    for name in ("c_t_mean", "c_q_mean", "hybrid_score_mean"):
        summary["fusion_selected_top7_profile_weighted"][name] = distribution(row["fusion_selected_top7_profile_weighted"][name] for row in run_rows)
    for name in ("engage_rate", "convert_rate"):
        summary["event_rates_profile_weighted"][name] = distribution(row["event_rates_profile_weighted"][name] for row in run_rows)
    for name in ("topsis_efficiency", "topsis_reconstruction", "q_component", "q_affine", "hybrid", "source_q_replay"):
        summary["diagnostic_max_abs_error"][name] = max(row["diagnostic_max_abs_error"][name] for row in run_rows)
    return summary, run_rows


def f1_at_7(rank: np.ndarray, truth: Iterable[int]) -> float:
    return len(set(int(x) for x in rank) & set(int(x) for x in truth)) / 7.0


def ndcg_at_7(rank: np.ndarray, truth: Iterable[int]) -> float:
    truth_set = set(int(x) for x in truth)
    dcg = sum((1.0 if int(item) in truth_set else 0.0) / math.log2(position + 2.0) for position, item in enumerate(rank))
    ideal = sum(1.0 / math.log2(position + 2.0) for position in range(7))
    return dcg / ideal


def lambda_posthoc_output() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    static_records = load_jsonl(STATIC / "outputs/canonical_main/main_catalogs.jsonl")
    xai_records = load_jsonl(XAI / "outputs/canonical/xai_attributions.jsonl")
    require(len(static_records) == len(xai_records) == 50, "Lambda source count mismatch")
    arm_id = ANCHORS["primary_corrected"]
    profile_order = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]
    grid = [round(value / 10.0, 1) for value in range(1, 10)]
    rows: list[dict[str, Any]] = []
    for lambda_q in grid:
        lambda_t = 1.0 - lambda_q
        by_profile = {profile: {"f1_at_7": [], "ndcg_at_7": []} for profile in profile_order}
        overall = {"f1_at_7": [], "ndcg_at_7": []}
        for run_index, (static_row, xai_row) in enumerate(zip(static_records, xai_records)):
            require(static_row["run_index"] == xai_row["run_index"] == run_index, "Lambda run order mismatch")
            run_values = {"f1_at_7": [], "ndcg_at_7": []}
            static_profiles = static_row["arms"][arm_id]["profiles"]
            for profile_index, profile_name in enumerate(profile_order):
                xai_profile = xai_row["profiles"][profile_index]
                require(xai_profile["profile_name"] == static_profiles[profile_index]["profile_name"] == profile_name, "Lambda profile order mismatch")
                norm_t = 2.0 * np.asarray(xai_profile["c_t"], dtype=float)
                norm_q = 2.0 * np.asarray(xai_profile["c_q"], dtype=float)
                rank = np.argsort(lambda_t * norm_t + lambda_q * norm_q)[::-1][:7]
                truth = static_profiles[profile_index]["gt_rank"]
                metrics = {"f1_at_7": f1_at_7(rank, truth), "ndcg_at_7": ndcg_at_7(rank, truth)}
                for metric, value in metrics.items():
                    by_profile[profile_name][metric].append(value)
                    run_values[metric].append(value)
            for metric in overall:
                overall[metric].append(float(np.mean(run_values[metric])))
        for profile_name in profile_order + ["all_profiles_mean_within_run"]:
            vectors = overall if profile_name.startswith("all_profiles") else by_profile[profile_name]
            for metric, vector in vectors.items():
                stats = bootstrap_summary(vector, f"lambda_posthoc|lambda_q={lambda_q:.1f}|{profile_name}|{metric}")
                rows.append({
                    "diagnostic_status": "evaluation_only_posthoc_oracle_not_for_tuning",
                    "lambda_q": lambda_q, "lambda_t": lambda_t, "profile": profile_name, "metric": metric, **stats,
                })
    summary = {
        "schema_version": "hre.lambda_posthoc_diagnostic.v1",
        "status": "evaluation_only_posthoc_oracle_not_for_tuning",
        "prespecified_primary_lambda_q": 0.5,
        "prespecified_primary_lambda_t": 0.5,
        "complete_grid_lambda_q": grid,
        "rank_definition": "numpy.argsort(score)[::-1][:7]",
        "unit": "catalog-resample; five fixed profiles averaged within run for overall rows",
        "rows": len(rows),
        "interpretation_boundary": "All fixed grid points are reported. This evaluation-label diagnostic cannot select, tune, or replace the prespecified 0.50/0.50 primary result.",
    }
    return summary, rows


def output_hash_rows(names: Iterable[str]) -> dict[str, str]:
    return {name: sha256(OUTPUT / name) for name in names}


def main() -> None:
    manifest = verify_run_manifest()
    attestations = verify_sources()
    require(not OUTPUT.exists(), f"Refusing nonempty/existing output directory: {OUTPUT}")

    static_payload, arm_rows, anchor_rows, contrast_rows, diagnostic_rows = static_outputs()
    drift_payload, drift_rows = drift_outputs()
    latency_payload = latency_output()
    xai_payload, xai_run_rows = xai_outputs()
    lambda_payload, lambda_rows = lambda_posthoc_output()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUTPUT / "static_all_arms.csv", ["metric", "arm_id", "candidate", "gt_bonus", "reward_model", "role", "method", "mean", "sample_sd", "ci95_lo", "ci95_hi", "n_catalogs"], arm_rows)
    write_csv(OUTPUT / "static_locked_anchors.csv", ["anchor", "metric", "row_type", "name", "arm_id", "mean", "sample_sd", "ci95_lo", "ci95_hi", "n_catalogs", "wins", "ties", "losses", "paired_t_stat", "paired_t_p_two_sided", "wilcoxon_stat", "wilcoxon_p_two_sided", "cohen_dz", "cohen_dz_defined"], anchor_rows)
    write_csv(OUTPUT / "static_all_contrasts.csv", ["family", "metric", "contrast_id", "method", "factor", "direction", "arm_a", "arm_b", "mean", "sample_sd", "ci95_lo", "ci95_hi", "n_catalogs", "wins", "ties", "losses", "paired_t_stat", "paired_t_p_two_sided", "wilcoxon_stat", "wilcoxon_p_two_sided", "cohen_dz", "cohen_dz_defined"], contrast_rows)
    write_csv(OUTPUT / "static_anchor_diagnostics.csv", ["anchor", "row_type", "profile", "metric", "checkpoint", "method", "mean", "sample_sd", "ci95_lo", "ci95_hi", "n_catalog_resamples", "bootstrap_reps", "bootstrap_seed"], diagnostic_rows)
    write_csv(OUTPUT / "lambda_posthoc_diagnostic.csv", ["diagnostic_status", "lambda_q", "lambda_t", "profile", "metric", "mean", "sample_sd", "ci95_lo", "ci95_hi", "n_catalog_resamples", "bootstrap_reps", "bootstrap_seed"], lambda_rows)
    write_csv(OUTPUT / "drift_all_conditions.csv", ["condition_class", "scenario", "reward_model", "row_type", "name", "metric", "checkpoint", "mean", "sample_sd", "ci95_lo", "ci95_hi", "n_catalog_resamples", "wins", "ties", "losses", "raw_catalog_resample_vector_json"], drift_rows)
    dump_json(OUTPUT / "latency_summary.json", latency_payload)
    write_jsonl(OUTPUT / "xai_run_level.jsonl", xai_run_rows)
    dump_json(OUTPUT / "xai_summary.json", xai_payload)

    source_manifest = {
        "schema_version": "hre.verified_science_extraction.source_hash_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": CREATED_AT,
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": OPERATION_ID,
        "extraction_run_manifest_sha256": sha256(HERE / "RUN_MANIFEST.json"),
        "upstream_files": manifest["upstream_files"],
        "terminal_attestations": {
            "static_overlay": {"full_sha256": sha256(OVERLAY / "outputs/canonical_overlay/FULL_VERIFICATION.json"), "status": attestations["static_overlay_full"]["status"], "verdict": attestations["static_overlay_full"]["verdict"]},
            "drift": {"full_sha256": sha256(DRIFT / "outputs/canonical/FULL_VERIFICATION.json"), "status": attestations["drift_full"]["status"]},
            "latency": {"full_sha256": sha256(LATENCY / "outputs/canonical/FULL_VERIFICATION.json"), "status": attestations["latency_full"]["status"], "verdict": attestations["latency_full"]["verdict"]},
            "xai": {"full_sha256": sha256(XAI / "outputs/canonical/FULL_VERIFICATION.json"), "status": attestations["xai_full"]["status"], "verdict": attestations["xai_full"]["verdict"]},
        },
    }
    dump_json(OUTPUT / "SOURCE_HASH_MANIFEST.json", source_manifest)

    derived_names = ["static_all_arms.csv", "static_locked_anchors.csv", "static_all_contrasts.csv", "static_anchor_diagnostics.csv", "lambda_posthoc_diagnostic.csv", "drift_all_conditions.csv", "latency_summary.json", "xai_run_level.jsonl", "xai_summary.json", "SOURCE_HASH_MANIFEST.json"]
    payload = {
        "schema_version": "hre.verified_scientific_payload.v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": CREATED_AT,
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": OPERATION_ID,
        "status": "extracted_pending_independent_verification",
        "completeness": {
            "static_arms": len(arm_rows) // 10,
            "static_arm_metric_method_rows": len(arm_rows),
            "static_anchor_rows": len(anchor_rows),
            "static_contrast_rows": len(contrast_rows),
            "static_anchor_diagnostic_rows": len(diagnostic_rows),
            "lambda_posthoc_rows": len(lambda_rows),
            "drift_condition_rows": len(drift_rows),
            "xai_run_rows": len(xai_run_rows),
        },
        "interpretation_boundaries": {
            "static_and_drift_unit": "paired catalog-resample/Monte Carlo run; five fixed profiles averaged within run; diagnostic/exploratory",
            "latency": latency_payload["claim_boundary"],
            "xai": xai_payload["interpretation_boundary"],
        },
        "static": static_payload,
        "lambda_posthoc_diagnostic": lambda_payload,
        "drift": drift_payload,
        "latency": latency_payload,
        "exact_xai": xai_payload,
        "derived_output_hashes": output_hash_rows(derived_names),
    }
    dump_json(OUTPUT / "VERIFIED_SCIENTIFIC_PAYLOAD.json", payload)
    print(f"EXTRACTED arms={len(arm_rows)} anchors={len(anchor_rows)} contrasts={len(contrast_rows)} anchor_diagnostics={len(diagnostic_rows)} lambda={len(lambda_rows)} drift={len(drift_rows)} xai_runs={len(xai_run_rows)}")
    print(f"PAYLOAD_SHA256 {sha256(OUTPUT / 'VERIFIED_SCIENTIFIC_PAYLOAD.json')}")


if __name__ == "__main__":
    main()
