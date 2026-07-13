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
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs/canonical"
STATIC = ROOT / "experiments/2026-07-12_codex_local_same_target_bridge_r01_producerenv/outputs/canonical_main"
DRIFT = ROOT / "experiments/2026-07-12_codex_local_same_target_drift_r01_producerenv/outputs/canonical"
LATENCY = ROOT / "experiments/2026-07-13_codex_local_latency_r1_producerenv/outputs/canonical"
XAI = ROOT / "experiments/2026-07-13_codex_local_exact_xai_r1_producerenv/outputs/canonical"
OVERLAY = ROOT / "experiments/2026-07-13_codex_static_verifier_topsis_gatefix/outputs/canonical_overlay"
PRIMARY = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
EXACT_R0 = "candidate=oracle_gt_hidden30__bonus=0.20__reward=implemented_r0"
PROFILE_ORDER = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value: str) -> None:
    raise VerificationError(f"Non-finite JSON constant: {value}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)
    require(isinstance(value, dict), f"Bad JSON root: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(bool(line.strip()), f"Blank JSONL line: {path}:{number}")
        value = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_object)
        require(isinstance(value, dict), f"Bad JSONL record: {path}:{number}")
        result.append(value)
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: Any, expected: Any, label: str, tolerance: float = 1e-12) -> None:
    if actual in ("", None) and expected in ("", None):
        return
    try:
        x, y = float(actual), float(expected)
    except (TypeError, ValueError):
        require(actual == expected, f"Value mismatch {label}: {actual!r} != {expected!r}")
        return
    require(math.isfinite(x) and math.isfinite(y) and abs(x - y) <= tolerance, f"Numeric mismatch {label}: {x} != {y}")


def sample_sd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def boot(values: Iterable[float], label: str) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    require(arr.size > 0 and np.all(np.isfinite(arr)), f"Invalid bootstrap source: {label}")
    seed = int.from_bytes(hashlib.sha256(f"verified_extraction_v1|{label}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(20000, len(arr)))].mean(axis=1)
    return {
        "mean": float(arr.mean()), "sample_sd": sample_sd(arr),
        "ci95_lo": float(np.quantile(means, 0.025)), "ci95_hi": float(np.quantile(means, 0.975)),
        "n_catalog_resamples": len(arr), "bootstrap_reps": 20000, "bootstrap_seed": seed,
    }


def verify_source_and_derived_hashes() -> dict[str, Any]:
    source = load_json(OUT / "SOURCE_HASH_MANIFEST.json")
    require(source.get("campaign_id") == CAMPAIGN_ID, "Source manifest campaign mismatch")
    require(source.get("extraction_run_manifest_sha256") == sha256(HERE / "RUN_MANIFEST.json"), "Extraction manifest link mismatch")
    for row in source["upstream_files"]:
        path = ROOT / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"], f"Upstream changed: {path}")
    attest = source["terminal_attestations"]
    require(attest["static_overlay"]["full_sha256"] == sha256(OVERLAY / "FULL_VERIFICATION.json") and attest["static_overlay"]["verdict"] == "PASS", "Static attestation mismatch")
    require(attest["drift"]["full_sha256"] == sha256(DRIFT / "FULL_VERIFICATION.json") and attest["drift"]["status"] == "PASS", "Drift attestation mismatch")
    require(attest["latency"]["full_sha256"] == sha256(LATENCY / "FULL_VERIFICATION.json") and attest["latency"]["verdict"] == "PASS", "Latency attestation mismatch")
    require(attest["xai"]["full_sha256"] == sha256(XAI / "FULL_VERIFICATION.json") and attest["xai"]["verdict"] == "PASS", "XAI attestation mismatch")
    payload = load_json(OUT / "VERIFIED_SCIENTIFIC_PAYLOAD.json")
    require(payload.get("campaign_id") == CAMPAIGN_ID and payload.get("status") == "extracted_pending_independent_verification", "Payload state mismatch")
    for name, expected in payload["derived_output_hashes"].items():
        require(sha256(OUT / name) == expected, f"Derived hash mismatch: {name}")
    return payload


def verify_static() -> int:
    terminal = load_json(STATIC / "main_results.json")
    analysis = terminal["analysis"]
    summaries = analysis["summaries"]
    raw = load_jsonl(STATIC / "main_catalogs.jsonl")
    arms = read_csv(OUT / "static_all_arms.csv")
    require(len(arms) == 200, "Static all-arm row count mismatch")
    for row in arms:
        summary = summaries[row["arm_id"]]
        metric = summary if row["metric"] == "f1_at_7" else summary["ndcg_at_7"]
        stats = metric["methods"][row["method"]]
        for key in ("mean", "sample_sd", "bootstrap_ci95_lo", "bootstrap_ci95_hi", "n_catalogs"):
            csv_key = {"bootstrap_ci95_lo": "ci95_lo", "bootstrap_ci95_hi": "ci95_hi"}.get(key, key)
            close(row[csv_key], stats[key], f"static arms {row['arm_id']} {key}")

    anchors = read_csv(OUT / "static_locked_anchors.csv")
    require(len(anchors) == 50, "Static anchor row count mismatch")
    anchor_map = {
        "exact_r0_reproduction": EXACT_R0,
        "internal_component_repair": "candidate=oracle_gt_hidden30__bonus=0.20__reward=component_continuous_fix",
        "primary_corrected": PRIMARY,
        "literal_fix_sensitivity": "candidate=full_catalog__bonus=0.00__reward=inclusive_range_fix",
        "historical_funnel_sensitivity": "candidate=full_catalog__bonus=0.00__reward=historical_funnel_coefficients_on_may_h",
    }
    for row in anchors:
        summary = summaries[anchor_map[row["anchor"]]]
        metric = summary if row["metric"] == "f1_at_7" else summary["ndcg_at_7"]
        stats = metric["methods"][row["name"]] if row["row_type"] == "method" else metric[row["name"]]
        close(row["mean"], stats["mean"], "anchor mean")
        close(row["sample_sd"], stats["sample_sd"], "anchor sd")
        close(row["ci95_lo"], stats["bootstrap_ci95_lo"], "anchor ci lo")
        close(row["ci95_hi"], stats["bootstrap_ci95_hi"], "anchor ci hi")

    contrasts = read_csv(OUT / "static_all_contrasts.csv")
    require(len(contrasts) == 188, "Static contrast row count mismatch")
    for row in contrasts:
        stats = analysis[row["family"]][row["metric"]][row["contrast_id"]][row["method"]]
        close(row["mean"], stats["mean"], "contrast mean")
        close(row["ci95_lo"], stats["bootstrap_ci95_lo"], "contrast lo")
        close(row["ci95_hi"], stats["bootstrap_ci95_hi"], "contrast hi")

    diagnostics = read_csv(OUT / "static_anchor_diagnostics.csv")
    require(len(diagnostics) == 310, "Static diagnostic row count mismatch")
    for row in diagnostics:
        arm_id = PRIMARY if row["anchor"] == "primary_corrected" else EXACT_R0
        method, metric = row["method"], row["metric"]
        if row["row_type"] == "checkpoint_overall":
            vector = [float(np.mean([p["checkpoint_f1"][row["checkpoint"]][method] for p in record["arms"][arm_id]["profiles"]])) for record in raw]
            label = f"{row['anchor']}|checkpoint_overall|{row['checkpoint']}|{method}"
        elif row["row_type"] == "checkpoint_profile_descriptive":
            pi = PROFILE_ORDER.index(row["profile"])
            vector = [record["arms"][arm_id]["profiles"][pi]["checkpoint_f1"][row["checkpoint"]][method] for record in raw]
            label = f"{row['anchor']}|checkpoint_profile|{row['profile']}|{row['checkpoint']}|{method}"
        else:
            pi = PROFILE_ORDER.index(row["profile"])
            vector = [record["arms"][arm_id]["profiles"][pi]["final_metrics"][method][metric] for record in raw]
            label = f"{row['anchor']}|final_profile|{row['profile']}|{metric}|{method}"
        expected = boot(vector, label)
        for key in expected:
            close(row[key], expected[key], f"static diagnostic {label} {key}")
    return len(arms) + len(anchors) + len(contrasts) + len(diagnostics)


def f1(rank: np.ndarray, truth: Iterable[int]) -> float:
    return len(set(map(int, rank)) & set(map(int, truth))) / 7.0


def ndcg(rank: np.ndarray, truth: Iterable[int]) -> float:
    truth_set = set(map(int, truth))
    ideal = sum(1.0 / math.log2(i + 2.0) for i in range(7))
    return sum((int(item) in truth_set) / math.log2(i + 2.0) for i, item in enumerate(rank)) / ideal


def verify_lambda() -> int:
    rows = read_csv(OUT / "lambda_posthoc_diagnostic.csv")
    require(len(rows) == 108, "Lambda row count mismatch")
    static = load_jsonl(STATIC / "main_catalogs.jsonl")
    xai = load_jsonl(XAI / "xai_attributions.jsonl")
    for row in rows:
        lq, lt = float(row["lambda_q"]), float(row["lambda_t"])
        by_run = []
        for sr, xr in zip(static, xai):
            values = []
            for pi, profile_name in enumerate(PROFILE_ORDER):
                if row["profile"] not in (profile_name, "all_profiles_mean_within_run"):
                    continue
                xp = xr["profiles"][pi]
                rank = np.argsort(lt * (2.0 * np.asarray(xp["c_t"])) + lq * (2.0 * np.asarray(xp["c_q"])))[::-1][:7]
                truth = sr["arms"][PRIMARY]["profiles"][pi]["gt_rank"]
                values.append(f1(rank, truth) if row["metric"] == "f1_at_7" else ndcg(rank, truth))
            by_run.append(float(np.mean(values)))
        label = f"lambda_posthoc|lambda_q={lq:.1f}|{row['profile']}|{row['metric']}"
        expected = boot(by_run, label)
        for key in expected:
            close(row[key], expected[key], f"lambda {label} {key}")
    return len(rows)


def normalized_auc(values: Iterable[float], grid: list[int]) -> float:
    return float(np.trapz(np.asarray(list(values), dtype=float), np.asarray(grid, dtype=float)) / (grid[-1] - grid[0]))


def verify_drift() -> int:
    rows = read_csv(OUT / "drift_all_conditions.csv")
    require(len(rows) == 72, "Drift row count mismatch")
    terminal = load_json(DRIFT / "TERMINAL.json")
    records = load_jsonl(DRIFT / "sealed_records.jsonl")
    for row in rows:
        if row["condition_class"] == "future_blind":
            stats = terminal["analysis"][row["scenario"]][row["reward_model"]][row["name"]][row["metric"]]
            close(row["mean"], stats["mean"], "drift mean")
            close(row["sample_sd"], stats["sample_sd"], "drift sd")
            close(row["ci95_lo"], stats["bootstrap_ci95"][0], "drift lo")
            close(row["ci95_hi"], stats["bootstrap_ci95"][1], "drift hi")
            if row["row_type"] == "paired_difference":
                require(json.loads(row["raw_catalog_resample_vector_json"]) == stats["raw_catalog_resample_vector"], "Drift paired vector mismatch")
        else:
            run_count = 50 if row["scenario"] == "sudden" else 30
            vector = [float(np.mean([p["checkpoints"][row["checkpoint"]]["f1"][row["name"]] for p in record[row["scenario"]]["legacy_exact"]])) for record in records[:run_count]]
            close(row["mean"], np.mean(vector), "legacy mean")
            close(row["sample_sd"], sample_sd(vector), "legacy sd")
            require(json.loads(row["raw_catalog_resample_vector_json"]) == vector, "Legacy vector mismatch")
    return len(rows)


def raw_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mean, std = float(arr.mean()), float(arr.std(ddof=0))
    return {
        "n": int(arr.size), "mean_ns": mean, "std_ns": std, "cv": std / mean,
        "min_ns": float(arr.min()), "median_ns": float(np.percentile(arr, 50, method="linear")),
        "p95_ns": float(np.percentile(arr, 95, method="linear")), "p99_ns": float(np.percentile(arr, 99, method="linear")), "max_ns": float(arr.max()),
    }


def verify_latency() -> int:
    summary = load_json(OUT / "latency_summary.json")
    source = load_json(LATENCY / "latency_results.json")
    durations = np.load(LATENCY / "raw_durations_ns.npy", allow_pickle=False)
    expected = raw_summary(durations)
    for key, value in expected.items():
        close(summary["raw_all_samples"][key], value, f"latency pooled {key}", 1e-9)
        close(source["analysis"]["raw_all_samples"][key], value, f"latency source pooled {key}", 1e-9)
    require(len(summary["passes_all_samples"]) == durations.shape[0] == 3, "Latency pass count mismatch")
    for pi, row in enumerate(summary["passes_all_samples"]):
        expected = raw_summary(durations[pi])
        for key, value in expected.items():
            close(row["all_samples"][key], value, f"latency pass {pi} {key}", 1e-9)
    require(summary["retention_gate"]["status"] == "PASS_RETAIN_CACHED_PATH_CLAIM", "Latency gate not retained")
    return int(durations.size)


def xai_run_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    shapley = np.asarray(row["topsis"]["shapley_values_normalized"], dtype=float)
    selected = []
    components = {name: [] for name in ("base", "engage", "convert")}
    ct, cq, hybrid = [], [], []
    profiles = []
    for profile in row["profiles"]:
        idx = np.asarray(profile["hybrid_top7_rank"], dtype=int)
        selected.append(shapley[idx])
        detail_components = {}
        for name in components:
            values = np.asarray(profile[f"c_q_{name}"])[idx]
            components[name].extend(values.tolist())
            detail_components[name] = {"signed_mean": float(values.mean()), "mean_abs": float(np.abs(values).mean())}
        ct_values, cq_values, h_values = np.asarray(profile["c_t"])[idx], np.asarray(profile["c_q"])[idx], np.asarray(profile["hybrid_score"])[idx]
        ct.extend(ct_values); cq.extend(cq_values); hybrid.extend(h_values)
        profiles.append({
            "profile_name": profile["profile_name"], "top7_rank": idx.tolist(),
            "reward_component_selected_top7": detail_components,
            "fusion_selected_top7": {"c_t_mean": float(ct_values.mean()), "c_q_mean": float(cq_values.mean()), "hybrid_score_mean": float(h_values.mean())},
            "engage_rate": profile["diagnostics"]["engage_count"] / 30000.0,
            "convert_rate": profile["diagnostics"]["convert_count"] / 30000.0,
        })
    selected_arr = np.concatenate(selected)
    return {
        "schema_version": "hre.exact_xai_run_level_summary.v1", "run_index": row["run_index"], "run_seed": row["run_seed"],
        "dataset_sha256": row["dataset_sha256"], "criteria": row["topsis"]["criteria"], "topsis_weights": row["topsis"]["weights"],
        "topsis_shapley_all_items": {"signed_mean": shapley.mean(axis=0).tolist(), "mean_abs": np.abs(shapley).mean(axis=0).tolist()},
        "topsis_shapley_selected_top7_profile_weighted": {"signed_mean": selected_arr.mean(axis=0).tolist(), "mean_abs": np.abs(selected_arr).mean(axis=0).tolist()},
        "topsis_shapley_selected_top7_hybrid_score_space": {"signed_mean": (0.5 * selected_arr.mean(axis=0)).tolist(), "mean_abs": (0.5 * np.abs(selected_arr).mean(axis=0)).tolist()},
        "reward_component_selected_top7_profile_weighted": {name: {"signed_mean": float(np.mean(values)), "mean_abs": float(np.mean(np.abs(values)))} for name, values in components.items()},
        "fusion_selected_top7_profile_weighted": {"c_t_mean": float(np.mean(ct)), "c_q_mean": float(np.mean(cq)), "hybrid_score_mean": float(np.mean(hybrid))},
        "event_rates_profile_weighted": {"engage_rate": float(np.mean([p["engage_rate"] for p in profiles])), "convert_rate": float(np.mean([p["convert_rate"] for p in profiles]))},
        "diagnostic_max_abs_error": {
            "topsis_efficiency": row["topsis"]["diagnostics"]["shapley_efficiency_max_abs_error"],
            "topsis_reconstruction": row["topsis"]["diagnostics"]["normalized_reconstruction_max_abs_error"],
            "q_component": max(p["diagnostics"]["q_component_raw_reconstruction_max_abs_error"] for p in row["profiles"]),
            "q_affine": max(p["diagnostics"]["c_q_affine_reconstruction_max_abs_error"] for p in row["profiles"]),
            "hybrid": max(p["diagnostics"]["hybrid_reconstruction_max_abs_error"] for p in row["profiles"]),
            "source_q_replay": max(p["diagnostics"]["source_q_replay_max_abs_error"] for p in row["profiles"]),
        },
        "profile_details_descriptive_only": profiles,
    }


def compare_tree(actual: Any, expected: Any, label: str) -> None:
    require(type(actual) is type(expected), f"Type mismatch {label}")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"Key mismatch {label}")
        for key in expected:
            compare_tree(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"List length mismatch {label}")
        for index, (x, y) in enumerate(zip(actual, expected)):
            compare_tree(x, y, f"{label}[{index}]")
    elif isinstance(expected, float):
        close(actual, expected, label, 1e-12)
    else:
        require(actual == expected, f"Value mismatch {label}")


def verify_xai() -> int:
    source = load_jsonl(XAI / "xai_attributions.jsonl")
    derived = load_jsonl(OUT / "xai_run_level.jsonl")
    require(len(source) == len(derived) == 50, "XAI derived row count mismatch")
    for index, (raw, actual) in enumerate(zip(source, derived)):
        compare_tree(actual, xai_run_summary(raw), f"xai_run[{index}]")
    summary = load_json(OUT / "xai_summary.json")
    require(summary["catalog_resamples"] == 50 and summary["profile_cells"] == 250, "XAI summary counts mismatch")
    require("not causal" in summary["interpretation_boundary"], "XAI boundary missing")
    # Independently validate each reported scalar distribution from the verified run rows.
    def dist(values: Iterable[float]) -> dict[str, Any]:
        arr = np.asarray(list(values), dtype=float)
        return {"n_catalog_resamples": len(arr), "mean": float(arr.mean()), "sample_sd": sample_sd(arr), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()), "p05": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95))}
    for ci, criterion in enumerate(summary["criteria"]):
        compare_tree(summary["topsis_weights"][criterion], dist(row["topsis_weights"][ci] for row in derived), f"xai_summary.weights.{criterion}")
        for section in ("topsis_shapley_all_items", "topsis_shapley_selected_top7_profile_weighted", "topsis_shapley_selected_top7_hybrid_score_space"):
            for statistic in ("signed_mean", "mean_abs"):
                compare_tree(summary[section][statistic][criterion], dist(row[section][statistic][ci] for row in derived), f"xai_summary.{section}.{statistic}.{criterion}")
    return len(derived)


def main() -> None:
    require(OUT.is_dir(), f"Missing extraction output: {OUT}")
    full_path = OUT / "FULL_EXTRACTION_VERIFICATION.json"
    require(not full_path.exists(), f"Refusing to overwrite terminal verification: {full_path}")
    payload = verify_source_and_derived_hashes()
    counts = {
        "static_rows_checked": verify_static(),
        "lambda_rows_checked": verify_lambda(),
        "drift_rows_checked": verify_drift(),
        "latency_raw_calls_checked": verify_latency(),
        "xai_run_rows_checked": verify_xai(),
    }
    require(payload["completeness"]["static_arm_metric_method_rows"] == 200, "Payload completeness mismatch")
    output_names = [p.name for p in OUT.iterdir() if p.is_file() and p.name != full_path.name]
    report = {
        "schema_version": "hre.verified_science_extraction.full_verification.v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "completed_verified",
        "verdict": "PASS",
        "verified_at": "2026-07-13T05:14:00+03:00",
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "operation_id": OPERATION_ID,
        "independence": "Standalone verifier; does not import extractor.py",
        "gates": [
            {"gate_id": "G01_source_hash_and_terminal_binding", "status": "PASS"},
            {"gate_id": "G02_all_static_arms_anchors_contrasts", "status": "PASS"},
            {"gate_id": "G03_convergence_profile_and_exact_r0_diagnostics", "status": "PASS"},
            {"gate_id": "G04_complete_lambda_grid_not_for_tuning", "status": "PASS"},
            {"gate_id": "G05_all_drift_conditions_and_legacy_reproduction", "status": "PASS"},
            {"gate_id": "G06_unfiltered_latency_raw_recomputation", "status": "PASS"},
            {"gate_id": "G07_predeclared_exact_xai_run_level_recomputation", "status": "PASS"},
            {"gate_id": "G08_output_hash_chain_and_interpretation_boundaries", "status": "PASS"},
        ],
        "counts": counts,
        "run_manifest_sha256": sha256(HERE / "RUN_MANIFEST.json"),
        "payload_sha256": sha256(OUT / "VERIFIED_SCIENTIFIC_PAYLOAD.json"),
        "output_hashes_before_full": {name: sha256(OUT / name) for name in sorted(output_names)},
    }
    full_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS checks={sum(counts.values())} FULL_SHA256={sha256(full_path)}")


if __name__ == "__main__":
    main()
