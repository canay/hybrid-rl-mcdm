from __future__ import annotations

# These variables are deliberately forced before NumPy (and therefore its BLAS
# runtime) is imported. The verifier checks both source ordering and the values
# recorded by the benchmark.
import os

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
THREAD_ENV_PREVIOUS = {key: os.environ.get(key) for key in THREAD_ENV_KEYS}
for _thread_key in THREAD_ENV_KEYS:
    os.environ[_thread_key] = "1"

import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib.util
import io
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CAMPAIGN_ROOT.parents[1]
CANONICAL_CONTRACT = {
    "warmup": 10_000,
    "passes": 3,
    "blocks": 20,
    "calls_per_block": 5_000,
    "bootstrap_replicates": 2_000,
    "timer_overhead_samples": 50_000,
}
SMOKE_DEFAULTS = {
    "warmup": 250,
    "passes": 1,
    "blocks": 2,
    "calls_per_block": 250,
    "bootstrap_replicates": 200,
    "timer_overhead_samples": 2_000,
}
BOOTSTRAP_SEED = 20_260_713
THRESHOLD_NS = 1_000_000
SOURCE_CAMPAIGN_NAME = "2026-07-12_codex_local_same_target_bridge_r01_producerenv"
EVIDENCE_CAMPAIGN_NAME = "2026-07-13_codex_static_verifier_topsis_gatefix"
CANONICAL_CATALOGS_SHA256 = "803eebfe09be8d62b5f446955f2106fe7ef8b220a979b68c9f9d71acb4827ecd"
CANONICAL_TERMINAL_SHA256 = "48677825f4446e2df427a0940dc8c0947b99aef1373ca5dfb6933f35728ad861"
CANONICAL_STATUS_SHA256 = "6470b05e83827637e34983511359a9eda24d26d0977d7976eb517dfe156ec2f3"
SOURCE_RUN_MANIFEST_SHA256 = "0428ecd9dc13f7241137d79428b47b94e03c9c41a2563978b25086adef1a2222"
OVERLAY_FULL_SHA256 = "a3112da73a28e9c68f3148b0a8668cc834472c7d5c765870ad1fed25e09fcd97"
OVERLAY_STATUS_SHA256 = "2513c8922a38e5f1a413d081b607225a55a620d7ad1cf540db208c48373b811f"
OVERLAY_RUN_MANIFEST_SHA256 = "ab29bb4782daf44c856f79ef8ad83559d4b59b637409476a43ed02dac8672ea7"
LOCKED_CORE_SHA256 = "46022b7348d7f0adcabeac8112009c53d82fa7669ef0b624a867c58794fc649f"
PRIMARY_ARM_ID = "candidate=full_catalog__bonus=0.00__reward=component_continuous_fix"
PROFILE_ORDER = ["budget", "quality_seeker", "explorer", "loyal", "balanced"]
RUN_INDICES = list(range(50))
PAIR_COUNT = 250
EXPECTED_PYTHON_VERSION = [3, 12, 12]
EXPECTED_PYTHON_IMPLEMENTATION = "CPython"
EXPECTED_NUMPY_VERSION = "1.26.0"
EXPECTED_TIMER_IMPLEMENTATION = "QueryPerformanceCounter()"
EXPECTED_CLAIM_BOUNDARY = "cached 400-item Q/TOPSIS static_hybrid_score plus full argsort top-7 only"


def utcish_local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def typed_array_contract(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    header = {"dtype": array.dtype.str, "shape": list(array.shape), "nbytes": int(array.nbytes)}
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return {**header, "typed_sha256": digest.hexdigest()}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(tmp_name, values, allow_pickle=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("latency_benchmark_locked_core", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load locked core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_config(args: argparse.Namespace) -> dict[str, int]:
    if args.mode == "canonical":
        for name in CANONICAL_CONTRACT:
            if getattr(args, name) is not None and getattr(args, name) != CANONICAL_CONTRACT[name]:
                raise AssertionError(f"Canonical {name} is immutable")
        return dict(CANONICAL_CONTRACT)
    config = dict(SMOKE_DEFAULTS)
    for name in config:
        if getattr(args, name) is not None:
            config[name] = int(getattr(args, name))
        if config[name] <= 0:
            raise AssertionError(f"{name} must be positive")
    return config


def expected_producer_executable() -> Path:
    return (
        PROJECT_ROOT
        / "experiments"
        / "_runtime"
        / "hre_submission_py312_numpy1260_pandas223"
        / "Scripts"
        / "python.exe"
    ).resolve()


def require_current_producer_environment() -> None:
    if list(sys.version_info[:3]) != EXPECTED_PYTHON_VERSION:
        raise AssertionError("Exact Python 3.12.12 producer environment required")
    if platform.python_implementation() != EXPECTED_PYTHON_IMPLEMENTATION:
        raise AssertionError("Exact CPython producer implementation required")
    if Path(sys.executable).resolve() != expected_producer_executable():
        raise AssertionError("Exact locked producer virtual-environment executable required")
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise AssertionError("Exact NumPy 1.26.0 producer environment required")
    timer = time.get_clock_info("perf_counter")
    if (
        timer.implementation != EXPECTED_TIMER_IMPLEMENTATION
        or timer.monotonic is not True
        or float(timer.resolution) <= 0
    ):
        raise AssertionError("Exact Windows QueryPerformanceCounter perf timer required")


def load_run_manifest() -> tuple[dict[str, Any], Path]:
    manifest_path = CAMPAIGN_ROOT / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "hre.latency.run_manifest.v1"
        or manifest.get("status") != "LOCKED"
        or manifest.get("campaign_id") != CAMPAIGN_ROOT.name
        or manifest.get("operation_id") != "HRE_R1_LATENCY_CANONICAL_LOCK_20260713_CODEX_20"
    ):
        raise AssertionError("Locked run manifest header mismatch")
    if manifest.get("execution_policy") != {
        "allowed_modes": ["smoke", "canonical"],
        "canonical_launch_authorized": True,
        "canonical_timing_executed_at_lock": False,
        "authorization_operation_id": "HRE_R1_LATENCY_CANONICAL_LOCK_20260713_CODEX_20",
    }:
        raise AssertionError("Locked execution policy mismatch")
    runner = manifest.get("lock_files", {}).get("runner", {})
    expected_runner = (CAMPAIGN_ROOT / "src" / "latency_benchmark.py").resolve()
    resolved_runner = (PROJECT_ROOT / Path(str(runner.get("path", "")))).resolve()
    if resolved_runner != expected_runner:
        raise AssertionError("Locked manifest runner path mismatch")
    if sha256_file(expected_runner) != runner.get("sha256"):
        raise AssertionError("Locked manifest runner hash mismatch")
    return manifest, manifest_path


def enforce_execution_policy(mode: str, run_manifest: dict[str, Any]) -> None:
    policy = run_manifest.get("execution_policy", {})
    if mode not in policy.get("allowed_modes", []):
        raise AssertionError("Requested mode is outside the locked execution allowlist")
    if mode == "canonical" and policy.get("canonical_launch_authorized") is not True:
        raise AssertionError("Locked policy does not authorize canonical timing")


def load_fixture(inputs_dir: Path):
    manifest_path = inputs_dir / "fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest_values = {
        "schema_version": "hre.latency_fixture.v2",
        "campaign_id": CAMPAIGN_ROOT.name,
        "fixture_file": "verified_vectors.npz",
        "lambda_q": 0.5,
        "top_k": 7,
    }
    for key, expected in required_manifest_values.items():
        if manifest.get(key) != expected:
            raise AssertionError(f"Fixture manifest contract mismatch: {key}")
    source = manifest.get("data_source", {})
    if (
        source.get("kind") != "canonical_scientific_payload"
        or source.get("campaign") != SOURCE_CAMPAIGN_NAME
        or source.get("mode") != "canonical"
        or source.get("arm_id") != PRIMARY_ARM_ID
        or source.get("run_indices") != RUN_INDICES
        or source.get("profile_order") != PROFILE_ORDER
        or source.get("main_catalogs_sha256") != CANONICAL_CATALOGS_SHA256
        or source.get("terminal_sha256") != CANONICAL_TERMINAL_SHA256
        or source.get("status_sha256") != CANONICAL_STATUS_SHA256
        or source.get("run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
    ):
        raise AssertionError("Canonical fixture data-source contract mismatch")
    expected_source = PROJECT_ROOT / "experiments" / SOURCE_CAMPAIGN_NAME
    source_paths = {
        "main_catalogs_path": expected_source / "outputs" / "canonical_main" / "main_catalogs.jsonl",
        "terminal_path": expected_source / "outputs" / "canonical_main" / "main_results.json",
        "status_path": expected_source / "outputs" / "canonical_main" / "status.json",
        "run_manifest_path": expected_source / "RUN_MANIFEST.json",
    }
    source_hash_keys = {
        "main_catalogs_path": CANONICAL_CATALOGS_SHA256,
        "terminal_path": CANONICAL_TERMINAL_SHA256,
        "status_path": CANONICAL_STATUS_SHA256,
        "run_manifest_path": SOURCE_RUN_MANIFEST_SHA256,
    }
    for key, expected_path in source_paths.items():
        resolved = (PROJECT_ROOT / Path(str(source.get(key, "")))).resolve()
        if resolved != expected_path.resolve() or sha256_file(resolved) != source_hash_keys[key]:
            raise AssertionError(f"Canonical data source path/hash mismatch: {key}")
    source_status = json.loads(source_paths["status_path"].read_text(encoding="utf-8"))
    if (
        source_status.get("status") != "completed_unverified"
        or source_status.get("mode") != "canonical"
        or source_status.get("campaign_id") != SOURCE_CAMPAIGN_NAME
        or source_status.get("terminal_sha256") != CANONICAL_TERMINAL_SHA256
        or source_status.get("run_manifest_sha256") != SOURCE_RUN_MANIFEST_SHA256
        or source_status.get("runs_completed") != 50
    ):
        raise AssertionError("Canonical data-source status mismatch")

    evidence = manifest.get("verification_evidence", {})
    if (
        evidence.get("kind") != "distinct_independent_canonical_overlay"
        or evidence.get("campaign") != EVIDENCE_CAMPAIGN_NAME
        or evidence.get("source_campaign") != SOURCE_CAMPAIGN_NAME
        or evidence.get("full_sha256") != OVERLAY_FULL_SHA256
        or evidence.get("status_sha256") != OVERLAY_STATUS_SHA256
        or evidence.get("run_manifest_sha256") != OVERLAY_RUN_MANIFEST_SHA256
        or evidence.get("status") != "completed_verified"
        or evidence.get("verdict") != "PASS"
    ):
        raise AssertionError("Distinct canonical evidence manifest contract mismatch")
    expected_evidence = PROJECT_ROOT / "experiments" / EVIDENCE_CAMPAIGN_NAME
    evidence_paths = {
        "full_path": expected_evidence / "outputs" / "canonical_overlay" / "FULL_VERIFICATION.json",
        "status_path": expected_evidence / "outputs" / "canonical_overlay" / "verification_status.json",
        "run_manifest_path": expected_evidence / "RUN_MANIFEST.json",
    }
    evidence_hashes = {
        "full_path": OVERLAY_FULL_SHA256,
        "status_path": OVERLAY_STATUS_SHA256,
        "run_manifest_path": OVERLAY_RUN_MANIFEST_SHA256,
    }
    for key, expected_path in evidence_paths.items():
        resolved = (PROJECT_ROOT / Path(str(evidence.get(key, "")))).resolve()
        if resolved != expected_path.resolve() or sha256_file(resolved) != evidence_hashes[key]:
            raise AssertionError(f"Distinct evidence path/hash mismatch: {key}")
    overlay_status = json.loads(evidence_paths["status_path"].read_text(encoding="utf-8"))
    overlay_full = json.loads(evidence_paths["full_path"].read_text(encoding="utf-8"))
    expected_outputs = {
        "main_catalogs.jsonl": CANONICAL_CATALOGS_SHA256,
        "main_results.json": CANONICAL_TERMINAL_SHA256,
        "status.json": CANONICAL_STATUS_SHA256,
    }
    if (
        overlay_status.get("status") != "completed_verified"
        or overlay_status.get("source_campaign_id") != SOURCE_CAMPAIGN_NAME
        or overlay_status.get("full_verification_sha256") != OVERLAY_FULL_SHA256
        or overlay_full.get("status") != "completed_verified"
        or overlay_full.get("verdict") != "PASS"
        or overlay_full.get("source_campaign_id") != SOURCE_CAMPAIGN_NAME
        or overlay_full.get("output_hashes") != expected_outputs
        or overlay_full.get("overlay_contract", {}).get("overlay_manifest_sha256") != OVERLAY_RUN_MANIFEST_SHA256
    ):
        raise AssertionError("Distinct overlay evidence content mismatch")

    if manifest.get("locked_core", {}).get("sha256") != LOCKED_CORE_SHA256:
        raise AssertionError("Fixture locked core allowlist mismatch")
    expected_core = expected_source / "src" / "original_hybrid_core.py"
    if (PROJECT_ROOT / Path(str(manifest.get("locked_core", {}).get("path", "")))).resolve() != expected_core.resolve():
        raise AssertionError("Fixture locked core path mismatch")
    if manifest.get("pair_schedule") != {
        "pair_count": 250,
        "order": "run_index major, PROFILE_ORDER minor",
        "canonical_calls_per_pair_per_block": 20,
        "canonical_calls_per_pair_total": 1200,
    }:
        raise AssertionError("Fixture pair schedule contract mismatch")
    fixture_path = inputs_dir / str(manifest["fixture_file"])
    if sha256_file(fixture_path) != manifest["fixture_sha256"]:
        raise AssertionError("Fixture file hash mismatch")
    core_path = PROJECT_ROOT / Path(str(manifest["locked_core"]["path"]))
    if sha256_file(core_path) != manifest["locked_core"]["sha256"]:
        raise AssertionError("Locked producer core hash mismatch")
    core = load_module(core_path)

    with np.load(fixture_path, allow_pickle=False) as data:
        if set(data.files) != {"topsis", "q_scores", "expected_top7"}:
            raise AssertionError("Fixture v2 array allowlist mismatch")
        topsis = np.asarray(data["topsis"]).copy()
        q_scores = np.asarray(data["q_scores"]).copy()
        expected_top7 = np.asarray(data["expected_top7"]).copy()
    expected_shapes = {"topsis": (50, 400), "q_scores": (50, 5, 400), "expected_top7": (50, 5, 7)}
    expected_dtypes = {"topsis": np.dtype("<f8"), "q_scores": np.dtype("<f8"), "expected_top7": np.dtype("<i8")}
    for name, array in {"topsis": topsis, "q_scores": q_scores, "expected_top7": expected_top7}.items():
        if array.shape != expected_shapes[name] or array.dtype != expected_dtypes[name]:
            raise AssertionError(f"Fixture v2 dtype/shape mismatch: {name}")
        if typed_array_contract(array) != manifest.get("arrays", {}).get(name):
            raise AssertionError(f"Fixture v2 typed hash mismatch: {name}")
    return manifest, fixture_path, core_path, core, q_scores, topsis, expected_top7


def _windows_cpu_name() -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        return None


def _windows_memory_bytes() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)


def set_normal_one_core(core_index: int | None) -> dict[str, Any]:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        kernel32.GetPriorityClass.restype = ctypes.c_ulong
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        allowed = int(process_mask.value)
        available = [idx for idx in range(allowed.bit_length()) if allowed & (1 << idx)]
        if not available:
            raise AssertionError("No available logical processor")
        selected = available[0] if core_index is None else int(core_index)
        if selected not in available:
            raise AssertionError(f"Requested core {selected} is outside current affinity mask")
        before_priority = int(kernel32.GetPriorityClass(handle))
        if not kernel32.SetPriorityClass(handle, 0x20):  # NORMAL_PRIORITY_CLASS
            raise ctypes.WinError(ctypes.get_last_error())
        selected_mask = 1 << selected
        if not kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(selected_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        after_process = ctypes.c_size_t()
        after_system = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(after_process), ctypes.byref(after_system)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        after_priority = int(kernel32.GetPriorityClass(handle))
        if int(after_process.value).bit_count() != 1 or after_priority != 0x20:
            raise AssertionError("Normal priority / one-core enforcement failed")
        return {
            "platform": "windows",
            "priority_class_before": before_priority,
            "priority_class_after": after_priority,
            "priority_name_after": "Normal",
            "affinity_mask_before": hex(allowed),
            "affinity_mask_after": hex(int(after_process.value)),
            "logical_processors_after": int(after_process.value).bit_count(),
            "selected_core_index": selected,
            "system_affinity_mask": hex(int(system_mask.value)),
        }

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise AssertionError("One-core affinity enforcement is unsupported on this platform")
    allowed_set = sorted(os.sched_getaffinity(0))
    selected = allowed_set[0] if core_index is None else int(core_index)
    if selected not in allowed_set:
        raise AssertionError(f"Requested core {selected} is outside current affinity set")
    os.sched_setaffinity(0, {selected})
    if len(os.sched_getaffinity(0)) != 1:
        raise AssertionError("One-core affinity enforcement failed")
    return {
        "platform": "posix",
        "priority_class_before": None,
        "priority_class_after": None,
        "priority_name_after": "Normal/default",
        "affinity_mask_before": allowed_set,
        "affinity_mask_after": sorted(os.sched_getaffinity(0)),
        "logical_processors_after": 1,
        "selected_core_index": selected,
    }


def read_runtime_state() -> dict[str, Any]:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        kernel32.GetPriorityClass.restype = ctypes.c_ulong
        handle = kernel32.GetCurrentProcess()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        priority = int(kernel32.GetPriorityClass(handle))
        mask = int(process_mask.value)
        return {
            "platform": "windows",
            "priority_class": priority,
            "priority_name": "Normal" if priority == 0x20 else f"non-Normal({priority})",
            "affinity_mask": hex(mask),
            "logical_processors": mask.bit_count(),
            "selected_core_indices": [idx for idx in range(mask.bit_length()) if mask & (1 << idx)],
            "system_affinity_mask": hex(int(system_mask.value)),
        }
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "platform": "posix",
        "priority_class": None,
        "priority_name": "Normal/default",
        "affinity_mask": affinity,
        "logical_processors": len(affinity),
        "selected_core_indices": affinity,
    }


def actual_threadpool_state() -> dict[str, Any]:
    try:
        from threadpoolctl import threadpool_info
    except Exception:
        return {
            "available": False,
            "provider": "threadpoolctl",
            "reason": "module unavailable in locked producer environment",
            "pools": [],
            "all_reported_threads_one": None,
        }
    pools = []
    for item in threadpool_info():
        pools.append(
            {
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "num_threads": item.get("num_threads"),
                "version": item.get("version"),
                "architecture": item.get("architecture"),
            }
        )
    reported = [int(item["num_threads"]) for item in pools if item.get("num_threads") is not None]
    return {
        "available": True,
        "provider": "threadpoolctl",
        "pools": pools,
        "all_reported_threads_one": bool(reported) and all(value == 1 for value in reported),
    }


def runtime_snapshot_ok(snapshot: dict[str, Any]) -> bool:
    return bool(
        snapshot.get("logical_processors") == 1
        and snapshot.get("priority_name") in {"Normal", "Normal/default"}
    )


def threadpool_snapshot_ok(snapshot: dict[str, Any]) -> bool:
    return bool(
        snapshot.get("all_reported_threads_one") is True
        if snapshot.get("available")
        else True
    )


def power_state() -> dict[str, Any]:
    if os.name != "nt":
        return {"active_scheme": None, "source": "not-windows"}
    try:
        completed = subprocess.run(
            ["powercfg", "/GETACTIVESCHEME"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return {
            "active_scheme": (completed.stdout or completed.stderr).strip(),
            "returncode": int(completed.returncode),
            "source": "powercfg /GETACTIVESCHEME",
        }
    except Exception as exc:
        return {"active_scheme": None, "source": "powercfg failed", "error": type(exc).__name__}


def numpy_runtime_text() -> dict[str, str]:
    config_buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Install `pyyaml` for better output")
        with contextlib.redirect_stdout(config_buffer):
            np.show_config()
    runtime_buffer = io.StringIO()
    if hasattr(np, "show_runtime"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Install `pyyaml` for better output")
            with contextlib.redirect_stdout(runtime_buffer):
                np.show_runtime()
    return {"show_config": config_buffer.getvalue(), "show_runtime": runtime_buffer.getvalue()}


def environment_record(
    runtime_enforcement: dict[str, Any],
    runtime_pre: dict[str, Any],
    threadpool_pre: dict[str, Any],
) -> dict[str, Any]:
    return {
        "recorded_at": utcish_local_now(),
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "libraries": {"numpy": np.__version__},
        "os": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "hardware": {
            "processor": platform.processor(),
            "windows_processor_name": _windows_cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "total_physical_memory_bytes": _windows_memory_bytes(),
        },
        "runtime_enforcement": runtime_enforcement,
        "runtime_pre": runtime_pre,
        "threadpool_pre": threadpool_pre,
        "thread_environment": {
            "forced_before_numpy_import": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
            "previous_values": THREAD_ENV_PREVIOUS,
        },
        "numpy_runtime": numpy_runtime_text(),
        "power": power_state(),
        "timer": {
            "name": "perf_counter_ns",
            "implementation": time.get_clock_info("perf_counter").implementation,
            "resolution_seconds": time.get_clock_info("perf_counter").resolution,
            "monotonic": time.get_clock_info("perf_counter").monotonic,
            "adjustable": time.get_clock_info("perf_counter").adjustable,
        },
    }


def summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise AssertionError("Cannot summarize an empty latency vector")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    return {
        "n": int(arr.size),
        "mean_ns": mean,
        "std_ns": std,
        "cv": float(std / mean) if mean > 0 else None,
        "min_ns": float(np.min(arr)),
        "median_ns": float(np.percentile(arr, 50, method="linear")),
        "p95_ns": float(np.percentile(arr, 95, method="linear")),
        "p99_ns": float(np.percentile(arr, 99, method="linear")),
        "max_ns": float(np.max(arr)),
    }


def block_bootstrap_ci(
    block_summaries: Sequence[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    if not block_summaries:
        raise AssertionError("No stable blocks are available for bootstrap")
    rng = np.random.RandomState(seed)
    metrics = ("median_ns", "p95_ns", "p99_ns")
    result: dict[str, Any] = {
        "method": "nonparametric bootstrap over blocks; estimator is the median of the selected block-level statistic",
        "replicates": int(replicates),
        "seed": int(seed),
        "n_blocks": len(block_summaries),
        "confidence_level": 0.95,
        "statistics": {},
    }
    for metric in metrics:
        values = np.asarray([float(block[metric]) for block in block_summaries], dtype=float)
        estimates = np.empty(replicates, dtype=float)
        for index in range(replicates):
            sample = values[rng.randint(0, len(values), size=len(values))]
            estimates[index] = float(np.median(sample))
        result["statistics"][metric] = {
            "estimate_ns": float(np.median(values)),
            "ci_lo_ns": float(np.percentile(estimates, 2.5, method="linear")),
            "ci_hi_ns": float(np.percentile(estimates, 97.5, method="linear")),
        }
    return result


def analyze_durations(
    durations: np.ndarray, calls_per_block: int, bootstrap_replicates: int
) -> dict[str, Any]:
    if durations.ndim != 3 or durations.shape[2] != calls_per_block:
        raise AssertionError("Raw duration shape does not match the block contract")
    passes, blocks, _ = durations.shape
    block_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    all_stable_blocks: list[dict[str, Any]] = []

    for pass_index in range(passes):
        pass_block_summaries = [summary(durations[pass_index, block]) for block in range(blocks)]
        medians = np.asarray([item["median_ns"] for item in pass_block_summaries], dtype=float)
        center = float(np.median(medians))
        mad = float(np.median(np.abs(medians - center)))
        tolerance = float(max(0.20 * center, 6.0 * 1.4826 * mad, 500.0))
        stable_indices: list[int] = []
        for block_index, block_summary in enumerate(pass_block_summaries):
            stable = bool(
                block_summary["n"] == calls_per_block
                and abs(float(block_summary["median_ns"]) - center) <= tolerance
            )
            row = {
                "pass_index": pass_index,
                "block_index": block_index,
                **block_summary,
                "stable": stable,
                "stability_center_ns": center,
                "stability_mad_ns": mad,
                "stability_tolerance_ns": tolerance,
            }
            block_rows.append(row)
            if stable:
                stable_indices.append(block_index)
                all_stable_blocks.append(row)
        stable_raw = np.concatenate([durations[pass_index, idx] for idx in stable_indices])
        pass_rows.append(
            {
                "pass_index": pass_index,
                "all_samples": summary(durations[pass_index].reshape(-1)),
                "stable_samples": summary(stable_raw),
                "stable_blocks": len(stable_indices),
                "total_blocks": blocks,
                "minimum_stable_blocks": int(math.ceil(0.90 * blocks)),
                "stability_sufficient": len(stable_indices) >= int(math.ceil(0.90 * blocks)),
                "bootstrap": block_bootstrap_ci(
                    [block_rows[pass_index * blocks + idx] for idx in stable_indices],
                    bootstrap_replicates,
                    BOOTSTRAP_SEED + pass_index + 1,
                ),
            }
        )

    stable_raw_all = np.concatenate(
        [durations[row["pass_index"], row["block_index"]] for row in all_stable_blocks]
    )
    all_stable_block_p99_under = all(float(row["p99_ns"]) < THRESHOLD_NS for row in all_stable_blocks)
    every_pass_stable_p99_under = all(
        float(pass_row["stable_samples"]["p99_ns"]) < THRESHOLD_NS for pass_row in pass_rows
    )
    stability_sufficient = all(bool(pass_row["stability_sufficient"]) for pass_row in pass_rows)
    pooled_raw_p99_under = float(summary(durations.reshape(-1))["p99_ns"]) < THRESHOLD_NS
    every_pass_all_p99_under = all(
        float(pass_row["all_samples"]["p99_ns"]) < THRESHOLD_NS for pass_row in pass_rows
    )
    return {
        "raw_all_samples": summary(durations.reshape(-1)),
        "stable_all_samples": summary(stable_raw_all),
        "blocks": block_rows,
        "passes": pass_rows,
        "bootstrap_all_stable_blocks": block_bootstrap_ci(
            all_stable_blocks, bootstrap_replicates, BOOTSTRAP_SEED
        ),
        "threshold_diagnostics": {
            "threshold_ns": THRESHOLD_NS,
            "primary_gate_basis": "all raw timed calls; no stability filtering",
            "pooled_raw_p99_strictly_below_threshold": pooled_raw_p99_under,
            "every_pass_all_sample_p99_strictly_below_threshold": every_pass_all_p99_under,
            "primary_condition_met": bool(pooled_raw_p99_under and every_pass_all_p99_under),
            "equality_to_threshold_fails": True,
            "stable_block_diagnostics_are_secondary": True,
            "stability_sufficient": stability_sufficient,
            "all_stable_block_p99_strictly_below_threshold": all_stable_block_p99_under,
            "every_pass_stable_sample_p99_strictly_below_threshold": every_pass_stable_p99_under,
        },
    }


def accuracy_checks(core, q_scores, topsis, expected_top7, manifest) -> list[dict[str, Any]]:
    rows = []
    for run_index in RUN_INDICES:
        for profile_index, profile_name in enumerate(PROFILE_ORDER):
            q = q_scores[run_index, profile_index]
            scores = core.static_hybrid_score(q, topsis[run_index], lambda_q=float(manifest["lambda_q"]))
            actual = [int(x) for x in np.argsort(scores)[::-1][: int(manifest["top_k"])]]
            expected = [int(x) for x in expected_top7[run_index, profile_index]]
            rows.append(
                {
                    "pair_index": run_index * len(PROFILE_ORDER) + profile_index,
                    "run_index": run_index,
                    "profile_index": profile_index,
                    "profile_name": profile_name,
                    "expected_top7": expected,
                    "actual_top7": actual,
                    "exact_match": actual == expected,
                }
            )
    return rows


def timer_overhead(samples: int) -> tuple[dict[str, Any], np.ndarray]:
    values = np.empty(samples, dtype=np.uint64)
    for index in range(samples):
        started = time.perf_counter_ns()
        finished = time.perf_counter_ns()
        values[index] = finished - started
    return ({"measurement": "back-to-back perf_counter_ns calls", **summary(values)}, values)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config = resolve_config(args)
    require_current_producer_environment()
    run_manifest, run_manifest_path = load_run_manifest()
    enforce_execution_policy(args.mode, run_manifest)
    if config["warmup"] % PAIR_COUNT != 0 or config["calls_per_block"] % PAIR_COUNT != 0:
        raise AssertionError("Warmup and each timed block must contain a balanced 250-pair schedule")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AssertionError("Output directory must be absent or empty (fail-closed)")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    started_wall = time.time()
    atomic_json(
        status_path,
        {
            "schema_version": "hre.latency_status.v1",
            "status": "starting",
            "mode": args.mode,
            "started_at": utcish_local_now(),
            "python_unbuffered_required": True,
        },
    )

    manifest, fixture_path, core_path, core, q_scores, topsis, expected_top7 = load_fixture(args.inputs_dir.resolve())
    q_pairs = q_scores.reshape(PAIR_COUNT, 400)
    topsis_pairs = np.repeat(topsis[:, None, :], len(PROFILE_ORDER), axis=1).reshape(PAIR_COUNT, 400)
    runtime_enforcement = set_normal_one_core(args.core_index)
    runtime_pre = read_runtime_state()
    threadpool_pre = actual_threadpool_state()
    environment = environment_record(runtime_enforcement, runtime_pre, threadpool_pre)
    if any(value != "1" for value in environment["thread_environment"]["forced_before_numpy_import"].values()):
        raise AssertionError("Thread environment was not forced to one")
    if not runtime_snapshot_ok(runtime_pre) or not threadpool_snapshot_ok(threadpool_pre):
        raise AssertionError("Pre-timing runtime/thread-pool contract failed")
    if not environment["timer"]["monotonic"] or float(environment["timer"]["resolution_seconds"]) <= 0:
        raise AssertionError("perf_counter timer must be monotonic with positive resolution")

    before_accuracy = accuracy_checks(core, q_scores, topsis, expected_top7, manifest)
    if len(before_accuracy) != PAIR_COUNT:
        raise AssertionError("Pre-benchmark accuracy must cover all 250 canonical pairs")
    if not all(row["exact_match"] for row in before_accuracy):
        raise AssertionError("Pre-benchmark accuracy mismatch")

    top_k = int(manifest["top_k"])
    lambda_q = float(manifest["lambda_q"])
    for call_index in range(config["warmup"]):
        pair_index = call_index % PAIR_COUNT
        scores = core.static_hybrid_score(q_pairs[pair_index], topsis_pairs[pair_index], lambda_q=lambda_q)
        np.argsort(scores)[::-1][:top_k]

    overhead, overhead_values = timer_overhead(config["timer_overhead_samples"])
    durations = np.empty(
        (config["passes"], config["blocks"], config["calls_per_block"]),
        dtype=np.uint64,
    )
    total_blocks = config["passes"] * config["blocks"]
    completed_blocks = 0
    global_call_index = 0
    for pass_index in range(config["passes"]):
        for block_index in range(config["blocks"]):
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                for call_index in range(config["calls_per_block"]):
                    pair_index = global_call_index % PAIR_COUNT
                    global_call_index += 1
                    started_ns = time.perf_counter_ns()
                    scores = core.static_hybrid_score(q_pairs[pair_index], topsis_pairs[pair_index], lambda_q=lambda_q)
                    np.argsort(scores)[::-1][:top_k]
                    finished_ns = time.perf_counter_ns()
                    durations[pass_index, block_index, call_index] = finished_ns - started_ns
            finally:
                if gc_was_enabled:
                    gc.enable()
            completed_blocks += 1
            elapsed = time.time() - started_wall
            eta = elapsed / completed_blocks * (total_blocks - completed_blocks)
            progress = 100.0 * completed_blocks / total_blocks
            atomic_json(
                status_path,
                {
                    "schema_version": "hre.latency_status.v1",
                    "status": "running",
                    "mode": args.mode,
                    "started_at": datetime.fromtimestamp(started_wall).astimezone().isoformat(timespec="seconds"),
                    "updated_at": utcish_local_now(),
                    "pass_index": pass_index,
                    "block_index": block_index,
                    "blocks_completed": completed_blocks,
                    "blocks_total": total_blocks,
                    "progress_percent": progress,
                    "eta_seconds": eta,
                    "stderr_health": "inspect wrapper stderr",
                    "python_unbuffered_required": True,
                },
            )
            print(
                f"progress={progress:.1f}% pass={pass_index + 1}/{config['passes']} "
                f"block={block_index + 1}/{config['blocks']} eta_seconds={eta:.1f}",
                flush=True,
            )

    after_accuracy = accuracy_checks(core, q_scores, topsis, expected_top7, manifest)
    if len(after_accuracy) != PAIR_COUNT:
        raise AssertionError("Post-benchmark accuracy must cover all 250 canonical pairs")
    if not all(row["exact_match"] for row in after_accuracy):
        raise AssertionError("Post-benchmark accuracy mismatch")
    runtime_post = read_runtime_state()
    threadpool_post = actual_threadpool_state()
    environment["runtime_post"] = runtime_post
    environment["threadpool_post"] = threadpool_post
    environment["thread_environment"]["post_timing_values"] = {
        key: os.environ.get(key) for key in THREAD_ENV_KEYS
    }
    environment["power_post"] = power_state()
    if not np.all(durations > 0) or not np.all(overhead_values > 0):
        raise AssertionError("All timed and timer-overhead durations must be positive")
    analysis = analyze_durations(durations, config["calls_per_block"], config["bootstrap_replicates"])
    accuracy_ok = all(row["exact_match"] for row in before_accuracy + after_accuracy)
    runtime_ok = (
        runtime_snapshot_ok(runtime_pre)
        and runtime_snapshot_ok(runtime_post)
        and runtime_pre["priority_class"] == runtime_post["priority_class"]
        and runtime_pre["affinity_mask"] == runtime_post["affinity_mask"]
        and threadpool_snapshot_ok(threadpool_pre)
        and threadpool_snapshot_ok(threadpool_post)
        and all(value == "1" for value in environment["thread_environment"]["forced_before_numpy_import"].values())
        and all(value == "1" for value in environment["thread_environment"]["post_timing_values"].values())
    )
    canonical_condition = bool(
        args.mode == "canonical"
        and analysis["threshold_diagnostics"]["primary_condition_met"]
        and accuracy_ok
        and runtime_ok
    )
    if args.mode == "canonical":
        gate_status = "PASS_RETAIN_CACHED_PATH_CLAIM" if canonical_condition else "FAIL_REMOVE_OR_NARROW_CLAIM"
    else:
        gate_status = "SMOKE_ONLY_NOT_CLAIMABLE"

    raw_path = output_dir / "raw_durations_ns.npy"
    overhead_path = output_dir / "timer_overhead_ns.npy"
    atomic_npy(raw_path, durations)
    atomic_npy(overhead_path, overhead_values)
    overhead["raw_artifact"] = {
        "path": overhead_path.name,
        "sha256": sha256_file(overhead_path),
        "dtype": str(overhead_values.dtype),
        "shape": list(overhead_values.shape),
    }
    result = {
        "schema_version": "hre.latency_result.v1",
        "campaign_id": CAMPAIGN_ROOT.name,
        "mode": args.mode,
        "status": "completed_unverified",
        "started_at": datetime.fromtimestamp(started_wall).astimezone().isoformat(timespec="seconds"),
        "completed_at": utcish_local_now(),
        "elapsed_seconds": time.time() - started_wall,
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
        "batch_size": 1,
        "config": config,
        "pair_schedule": {
            "pair_count": PAIR_COUNT,
            "order": "run_index major, PROFILE_ORDER minor",
            "calls_per_pair_per_block": config["calls_per_block"] // PAIR_COUNT,
            "calls_per_pair_total": (
                config["passes"] * config["blocks"] * config["calls_per_block"] // PAIR_COUNT
            ),
            "warmup_calls_per_pair": config["warmup"] // PAIR_COUNT,
        },
        "fixture": {
            "manifest_path": str((args.inputs_dir.resolve() / "fixture_manifest.json").relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(args.inputs_dir.resolve() / "fixture_manifest.json"),
            "fixture_path": str(fixture_path.relative_to(PROJECT_ROOT)),
            "fixture_sha256": sha256_file(fixture_path),
            "locked_core_path": str(core_path.relative_to(PROJECT_ROOT)),
            "locked_core_sha256": sha256_file(core_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "run_manifest": {
            "path": str(run_manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(run_manifest_path),
            "status": run_manifest["status"],
        },
        "environment": environment,
        "timer_overhead": overhead,
        "accuracy": {"before": before_accuracy, "after": after_accuracy, "all_exact": accuracy_ok},
        "raw_artifact": {
            "path": raw_path.name,
            "sha256": sha256_file(raw_path),
            "dtype": str(durations.dtype),
            "shape": list(durations.shape),
        },
        "analysis": analysis,
        "retention_gate": {
            "threshold_ns": THRESHOLD_NS,
            "status": gate_status,
            "canonical_condition_met": canonical_condition,
            "accuracy_ok": accuracy_ok,
            "runtime_ok": runtime_ok,
            "verifier_pass_required": True,
        },
    }
    result_path = output_dir / "latency_results.json"
    atomic_json(result_path, result)
    terminal_sha256 = sha256_file(result_path)
    atomic_json(
        status_path,
        {
            "schema_version": "hre.latency_status.v1",
            "status": "completed_unverified",
            "mode": args.mode,
            "completed_at": result["completed_at"],
            "progress_percent": 100.0,
            "terminal_path": result_path.name,
            "terminal_sha256": terminal_sha256,
            "raw_sha256": result["raw_artifact"]["sha256"],
            "stderr_health": "inspect wrapper stderr",
            "python_unbuffered_required": True,
        },
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--inputs-dir", type=Path, default=CAMPAIGN_ROOT / "inputs")
    result.add_argument("--mode", choices=("smoke", "canonical"), required=True)
    result.add_argument("--core-index", type=int)
    for name in CANONICAL_CONTRACT:
        result.add_argument("--" + name.replace("_", "-"), dest=name, type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        result = execute(args)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "mode": result["mode"],
                    "gate": result["retention_gate"]["status"],
                    "raw_sha256": result["raw_artifact"]["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        try:
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "RUN_FAILURE.json"
            atomic_json(
                failure_path,
                {
                    "schema_version": "hre.latency_run_failure.v1",
                    "status": "failed_closed",
                    "mode": args.mode,
                    "failed_at": utcish_local_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_sha256": hashlib.sha256(traceback.format_exc().encode("utf-8")).hexdigest(),
                },
            )
            atomic_json(
                output_dir / "status.json",
                {
                    "schema_version": "hre.latency_status.v1",
                    "status": "failed_closed",
                    "mode": args.mode,
                    "failed_at": utcish_local_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "terminal_path": failure_path.name,
                    "terminal_sha256": sha256_file(failure_path),
                },
            )
        finally:
            raise


if __name__ == "__main__":
    raise SystemExit(main())
