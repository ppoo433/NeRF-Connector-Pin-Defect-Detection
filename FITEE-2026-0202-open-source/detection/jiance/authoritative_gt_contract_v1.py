# -*- coding: utf-8 -*-
"""Shared authoritative GT contract helpers for the partitioned tolerance GT root."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_GT_PARTITION = "bad"
KNOWN_GT_PARTITIONS = ("bad", "normal")
SAMPLE_UID_SEP = "__"
_PARTITION_PREFERRED_PIN_NAME = {
    "bad": "{sample_id}_pin.ply",
    "normal": "{sample_id}_pins.ply",
}
_PARTITION_PIN_NAME_CANDIDATES = {
    "bad": ("{sample_id}_pin.ply", "{sample_id}.ply"),
    "normal": ("{sample_id}_pins.ply", "{sample_id}_pin.ply", "{sample_id}.ply"),
}

_LEGACY_PIN_RE = re.compile(r"^(?P<sample_id>\d+)$")
_PIN_RE = re.compile(r"^(?P<sample_id>\d+)_pin$")
_PINS_RE = re.compile(r"^(?P<sample_id>\d+)_pins$")
_BASE_RE = re.compile(r"^(?P<sample_id>\d+)_base$")


def _normalize_sample_id(sample_id: Any) -> str:
    text = str(sample_id or "").strip()
    if not text:
        raise ValueError("sample_id is required")
    if not text.isdigit():
        raise ValueError(f"sample_id must be numeric: {sample_id!r}")
    return text


def make_sample_uid(sample_id: Any, gt_partition: Any) -> str:
    sample_id_n = _normalize_sample_id(sample_id)
    partition = _normalize_gt_partition(gt_partition)
    return f"{partition}{SAMPLE_UID_SEP}{sample_id_n}"


def parse_sample_uid(sample_uid: Any) -> Tuple[str, str]:
    text = str(sample_uid or "").strip()
    if not text:
        raise ValueError("sample_uid is required")
    if SAMPLE_UID_SEP not in text:
        raise ValueError(f"invalid sample_uid: {sample_uid!r}")
    partition, sample_id = text.split(SAMPLE_UID_SEP, 1)
    return _normalize_gt_partition(partition), _normalize_sample_id(sample_id)


def _sample_id_from_any(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return text
    try:
        _partition, sample_id = parse_sample_uid(text)
        return sample_id
    except Exception:
        return ""


def _partition_from_path_text(path_like: Any) -> str:
    text = str(path_like or "").strip()
    if not text:
        return ""
    try:
        parts = [str(part).strip().lower() for part in Path(text).parts]
    except Exception:
        parts = [part.strip().lower() for part in text.replace("/", "\\").split("\\") if part.strip()]
    for idx, part in enumerate(parts[:-1]):
        if part in {"split", "tolerance_anomaly_gt"} and idx + 1 < len(parts) and parts[idx + 1] in KNOWN_GT_PARTITIONS:
            return str(parts[idx + 1])
    labels = [part for part in parts if part in KNOWN_GT_PARTITIONS]
    if len(set(labels)) == 1:
        return str(labels[0])
    return ""


def source_sample_id_from_manifest_row(manifest_row: Mapping[str, Any]) -> str:
    row = dict(manifest_row or {})
    for key in ("source_sample_id", "raw_sample_id", "numeric_sample_id"):
        sample_id = _sample_id_from_any(row.get(key, ""))
        if sample_id:
            return sample_id
    sample_id = _sample_id_from_any(row.get("sample_id", ""))
    if sample_id:
        return sample_id
    for key in ("gt_ply", "authoritative_gt_ply", "pin_ply", "base_ply"):
        text = str(row.get(key, "") or "").strip()
        if not text:
            continue
        try:
            parsed_sample_id, _kind = _parse_gt_filename(Path(text).name)
            return parsed_sample_id
        except Exception:
            stem = Path(text).stem
            head = str(stem).split("_", 1)[0].strip()
            if head.isdigit():
                return head
    raise ValueError(f"cannot resolve source_sample_id from manifest row: {row}")


def sample_uid_from_manifest_row(
    manifest_row: Mapping[str, Any],
    *,
    default_partition: str = DEFAULT_GT_PARTITION,
) -> str:
    row = dict(manifest_row or {})
    explicit = str(row.get("sample_uid", "") or "").strip()
    if explicit:
        partition, sample_id = parse_sample_uid(explicit)
        return make_sample_uid(sample_id, partition)
    partition = (
        str(row.get("gt_partition", "") or "").strip().lower()
        or _partition_from_path_text(row.get("pin_ply", ""))
        or _partition_from_path_text(row.get("base_ply", ""))
        or _partition_from_path_text(row.get("gt_ply", ""))
        or _partition_from_path_text(row.get("authoritative_gt_ply", ""))
        or str(default_partition or DEFAULT_GT_PARTITION).strip().lower()
    )
    return make_sample_uid(source_sample_id_from_manifest_row(row), partition)


def _resolve_gt_root(gt_root: Any) -> Path:
    path = Path(str(gt_root or "").strip()).resolve()
    if not str(path):
        raise ValueError("gt_root is required")
    return path


def _normalize_gt_partition(
    gt_partition: Any,
    *,
    default_partition: str = DEFAULT_GT_PARTITION,
    allow_all: bool = False,
) -> str:
    text = str(gt_partition or "").strip().lower()
    if not text:
        text = str(default_partition or DEFAULT_GT_PARTITION).strip().lower()
    if allow_all and text in {"all", "*"}:
        return "all"
    if text not in KNOWN_GT_PARTITIONS:
        raise ValueError(f"unsupported gt_partition: {gt_partition!r}")
    return text


def _natural_sample_key(value: Any) -> Tuple[int, str]:
    text = str(value or "").strip()
    return (len(text), text)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _preferred_partition_dir(gt_root: Path, gt_partition: str) -> Path:
    partition = _normalize_gt_partition(gt_partition)
    if partition == "bad":
        bad_dir = (gt_root / "bad").resolve()
        if bad_dir.is_dir():
            return bad_dir
        return gt_root.resolve()
    normal_dir = (gt_root / "normal").resolve()
    if normal_dir.is_dir():
        return normal_dir
    return gt_root.resolve()


def _partition_scan_dirs(gt_root: Path, gt_partition: str) -> List[Path]:
    partition = _normalize_gt_partition(gt_partition)
    if partition == "bad":
        bad_dir = (gt_root / "bad").resolve()
        if bad_dir.is_dir():
            return [bad_dir]
        return [gt_root.resolve()]
    normal_dir = (gt_root / "normal").resolve()
    if normal_dir.is_dir():
        return [normal_dir]
    return [gt_root.resolve()]


def _pin_candidate_names(sample_id: str, gt_partition: str, *, allow_legacy: bool) -> List[str]:
    partition = _normalize_gt_partition(gt_partition)
    names = [pattern.format(sample_id=sample_id) for pattern in _PARTITION_PIN_NAME_CANDIDATES[partition]]
    if not allow_legacy:
        names = [name for name in names if name != f"{sample_id}.ply"]
    deduped: List[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _pin_variant_from_name(name: str) -> str:
    stem = Path(str(name)).stem
    if _PINS_RE.fullmatch(stem):
        return "pins"
    if _PIN_RE.fullmatch(stem):
        return "pin"
    if _LEGACY_PIN_RE.fullmatch(stem):
        return "legacy"
    raise ValueError(f"unsupported pin filename: {name!r}")


def _parse_gt_filename(name: str) -> Tuple[str, str]:
    stem = Path(str(name or "")).stem
    for pattern, kind in ((_BASE_RE, "base"), (_PINS_RE, "pins"), (_PIN_RE, "pin"), (_LEGACY_PIN_RE, "legacy")):
        match = pattern.fullmatch(stem)
        if match:
            return str(match.group("sample_id")), str(kind)
    raise ValueError(f"unsupported authoritative GT filename: {name!r}")


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in list(paths or []):
        if path.is_file():
            return path
    return None


def canonical_authoritative_gt_pin_path(
    sample_id: Any,
    gt_root: Any,
    gt_partition: Any = DEFAULT_GT_PARTITION,
) -> Path:
    sample_id_n = _normalize_sample_id(sample_id)
    gt_root_p = _resolve_gt_root(gt_root)
    partition = _normalize_gt_partition(gt_partition)
    storage_dir = _preferred_partition_dir(gt_root_p, partition)
    filename = _PARTITION_PREFERRED_PIN_NAME[partition].format(sample_id=sample_id_n)
    return (storage_dir / filename).resolve()


def legacy_authoritative_gt_pin_path(
    sample_id: Any,
    gt_root: Any,
    gt_partition: Any = DEFAULT_GT_PARTITION,
) -> Path:
    sample_id_n = _normalize_sample_id(sample_id)
    gt_root_p = _resolve_gt_root(gt_root)
    partition = _normalize_gt_partition(gt_partition)
    storage_dir = _preferred_partition_dir(gt_root_p, partition)
    return (storage_dir / f"{sample_id_n}.ply").resolve()


def canonical_authoritative_gt_base_path(
    sample_id: Any,
    gt_root: Any,
    gt_partition: Any = DEFAULT_GT_PARTITION,
) -> Path:
    sample_id_n = _normalize_sample_id(sample_id)
    gt_root_p = _resolve_gt_root(gt_root)
    partition = _normalize_gt_partition(gt_partition)
    storage_dir = _preferred_partition_dir(gt_root_p, partition)
    return (storage_dir / f"{sample_id_n}_base.ply").resolve()


def infer_authoritative_gt_partition_from_path(
    path_like: Any,
    gt_root: Any,
    *,
    default_partition: str = DEFAULT_GT_PARTITION,
) -> str:
    path = Path(str(path_like or "").strip()).resolve()
    gt_root_p = _resolve_gt_root(gt_root)
    if not _is_relative_to(path, gt_root_p):
        raise ValueError(f"path is outside authoritative GT root: {path}")
    rel = path.relative_to(gt_root_p)
    if len(rel.parts) >= 2 and str(rel.parts[0]).lower() in KNOWN_GT_PARTITIONS:
        return str(rel.parts[0]).lower()
    return _normalize_gt_partition(default_partition)


def resolve_authoritative_gt_record(
    sample_id: Any,
    gt_root: Any,
    *,
    allow_legacy: bool = True,
    gt_partition: Any = DEFAULT_GT_PARTITION,
    gt_ply: Any = "",
    gt_base_ply: Any = "",
    default_partition: str = DEFAULT_GT_PARTITION,
) -> Dict[str, Any]:
    gt_root_p = _resolve_gt_root(gt_root)
    gt_ply_text = str(gt_ply or "").strip()
    gt_base_ply_text = str(gt_base_ply or "").strip()

    if gt_ply_text:
        pin_path = Path(gt_ply_text).resolve()
        if not _is_relative_to(pin_path, gt_root_p):
            raise ValueError(f"gt_ply is outside authoritative GT root: {pin_path}")
        parsed_sample_id, pin_variant = _parse_gt_filename(pin_path.name)
        if pin_variant == "base":
            raise ValueError(f"gt_ply cannot point to a base file: {pin_path}")
        sample_id_n = parsed_sample_id
        supplied_sample_id = str(sample_id or "").strip()
        if supplied_sample_id and _normalize_sample_id(supplied_sample_id) != sample_id_n:
            raise ValueError(
                "sample_id does not match gt_ply: "
                f"sample_id={supplied_sample_id!r} gt_ply={pin_path}"
            )
        partition = infer_authoritative_gt_partition_from_path(
            pin_path,
            gt_root_p,
            default_partition=str(gt_partition or default_partition or DEFAULT_GT_PARTITION),
        )
        if gt_base_ply_text:
            base_path = Path(gt_base_ply_text).resolve()
            if not _is_relative_to(base_path, gt_root_p):
                raise ValueError(f"gt_base_ply is outside authoritative GT root: {base_path}")
        else:
            base_path = (pin_path.parent / f"{sample_id_n}_base.ply").resolve()
        if not pin_path.is_file() or not base_path.is_file():
            fallback_record = resolve_authoritative_gt_record(
                sample_id_n,
                gt_root_p,
                allow_legacy=allow_legacy,
                gt_partition=partition,
                gt_ply="",
                gt_base_ply=gt_base_ply_text,
                default_partition=partition,
            )
            if not pin_path.is_file() and bool(fallback_record.get("pin_exists", False)):
                pin_path = Path(str(fallback_record.get("pin_path", ""))).resolve()
                pin_variant = str(fallback_record.get("pin_variant", pin_variant) or pin_variant)
            if not base_path.is_file() and bool(fallback_record.get("base_exists", False)):
                base_path = Path(str(fallback_record.get("base_path", ""))).resolve()
        return {
            "sample_id": str(sample_id_n),
            "gt_partition": str(partition),
            "gt_sample_key": f"{partition}:{sample_id_n}",
            "pin_path": str(pin_path),
            "base_path": str(base_path),
            "pin_variant": str(pin_variant),
            "pin_exists": bool(pin_path.is_file()),
            "base_exists": bool(base_path.is_file()),
            "storage_dir": str(pin_path.parent.resolve()),
            "resolution_source": "explicit_gt_ply",
        }

    sample_id_n = _normalize_sample_id(sample_id)
    partition = _normalize_gt_partition(gt_partition, default_partition=default_partition)
    scan_dirs = _partition_scan_dirs(gt_root_p, partition)
    pin_candidates: List[Path] = []
    for scan_dir in scan_dirs:
        pin_candidates.extend((scan_dir / name).resolve() for name in _pin_candidate_names(sample_id_n, partition, allow_legacy=allow_legacy))
    pin_path = _first_existing(pin_candidates) or canonical_authoritative_gt_pin_path(sample_id_n, gt_root_p, partition)
    pin_variant = _pin_variant_from_name(pin_path.name)
    if gt_base_ply_text:
        base_path = Path(gt_base_ply_text).resolve()
        if not _is_relative_to(base_path, gt_root_p):
            raise ValueError(f"gt_base_ply is outside authoritative GT root: {base_path}")
    else:
        base_path = canonical_authoritative_gt_base_path(sample_id_n, gt_root_p, partition)
    return {
        "sample_id": str(sample_id_n),
        "gt_partition": str(partition),
        "gt_sample_key": f"{partition}:{sample_id_n}",
        "pin_path": str(pin_path),
        "base_path": str(base_path),
        "pin_variant": str(pin_variant),
        "pin_exists": bool(pin_path.is_file()),
        "base_exists": bool(base_path.is_file()),
        "storage_dir": str(pin_path.parent.resolve()),
        "resolution_source": "sample_id_partition",
    }


def authoritative_gt_record_from_manifest_row(
    manifest_row: Mapping[str, Any],
    gt_root: Any,
    *,
    allow_legacy: bool = True,
    default_partition: str = DEFAULT_GT_PARTITION,
) -> Dict[str, Any]:
    row = dict(manifest_row or {})
    return resolve_authoritative_gt_record(
        row.get("sample_id", ""),
        gt_root,
        allow_legacy=allow_legacy,
        gt_partition=row.get("gt_partition", ""),
        gt_ply=row.get("gt_ply", ""),
        gt_base_ply=row.get("gt_base_ply", ""),
        default_partition=default_partition,
    )


def resolve_authoritative_gt_pin_path(
    sample_id: Any,
    gt_root: Any,
    allow_legacy: bool = True,
    gt_partition: Any = DEFAULT_GT_PARTITION,
    gt_ply: Any = "",
    gt_base_ply: Any = "",
) -> Path:
    record = resolve_authoritative_gt_record(
        sample_id,
        gt_root,
        allow_legacy=allow_legacy,
        gt_partition=gt_partition,
        gt_ply=gt_ply,
        gt_base_ply=gt_base_ply,
    )
    return Path(str(record["pin_path"])).resolve()


def resolve_authoritative_gt_base_path(
    sample_id: Any,
    gt_root: Any,
    allow_legacy: bool = True,
    gt_partition: Any = DEFAULT_GT_PARTITION,
    gt_ply: Any = "",
    gt_base_ply: Any = "",
) -> Path:
    record = resolve_authoritative_gt_record(
        sample_id,
        gt_root,
        allow_legacy=allow_legacy,
        gt_partition=gt_partition,
        gt_ply=gt_ply,
        gt_base_ply=gt_base_ply,
    )
    return Path(str(record["base_path"])).resolve()


def resolve_authoritative_gt_pair(
    sample_id: Any,
    gt_root: Any,
    allow_legacy: bool = True,
    gt_partition: Any = DEFAULT_GT_PARTITION,
    gt_ply: Any = "",
    gt_base_ply: Any = "",
) -> Tuple[Path, Path]:
    record = resolve_authoritative_gt_record(
        sample_id,
        gt_root,
        allow_legacy=allow_legacy,
        gt_partition=gt_partition,
        gt_ply=gt_ply,
        gt_base_ply=gt_base_ply,
    )
    return Path(str(record["pin_path"])).resolve(), Path(str(record["base_path"])).resolve()


def _scan_partition_inventory(
    gt_root: Path,
    gt_partition: str,
    *,
    allow_legacy: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    partition = _normalize_gt_partition(gt_partition)
    scan_dirs = _partition_scan_dirs(gt_root, partition)
    if not scan_dirs:
        return [], []
    storage_dir = scan_dirs[0]
    pin_by_sample_id: Dict[str, Dict[str, Path]] = {}
    base_by_sample_id: Dict[str, Path] = {}
    unexpected_files: List[str] = []

    if not storage_dir.is_dir():
        return [], []

    for path in storage_dir.iterdir():
        if not path.is_file() or str(path.suffix).lower() != ".ply":
            continue
        try:
            sample_id, kind = _parse_gt_filename(path.name)
        except ValueError:
            unexpected_files.append(str(path.name))
            continue
        if kind == "base":
            base_by_sample_id[str(sample_id)] = path.resolve()
            continue
        if kind == "legacy" and not allow_legacy:
            continue
        pin_slot = pin_by_sample_id.setdefault(str(sample_id), {})
        pin_slot[str(kind)] = path.resolve()

    sample_ids = sorted(set(pin_by_sample_id.keys()) | set(base_by_sample_id.keys()), key=_natural_sample_key)
    records: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        pin_candidates = [
            pin_by_sample_id.get(sample_id, {}).get(_pin_variant_from_name(name), None)
            for name in _pin_candidate_names(sample_id, partition, allow_legacy=allow_legacy)
        ]
        pin_path = _first_existing([path for path in pin_candidates if isinstance(path, Path)])
        if pin_path is None:
            pin_path = canonical_authoritative_gt_pin_path(sample_id, gt_root, partition)
            pin_variant = _pin_variant_from_name(pin_path.name)
        else:
            pin_variant = _pin_variant_from_name(pin_path.name)
        base_path = base_by_sample_id.get(sample_id, canonical_authoritative_gt_base_path(sample_id, gt_root, partition))
        records.append(
            {
                "sample_id": str(sample_id),
                "gt_partition": str(partition),
                "gt_sample_key": f"{partition}:{sample_id}",
                "pin_path": str(pin_path),
                "base_path": str(base_path),
                "pin_exists": bool(Path(pin_path).is_file()),
                "base_exists": bool(Path(base_path).is_file()),
                "pin_variant": str(pin_variant),
                "storage_dir": str(storage_dir),
                "canonical_pin_path": str(canonical_authoritative_gt_pin_path(sample_id, gt_root, partition)),
                "canonical_base_path": str(canonical_authoritative_gt_base_path(sample_id, gt_root, partition)),
                "legacy_pin_path": str(legacy_authoritative_gt_pin_path(sample_id, gt_root, partition)),
            }
        )
    return records, sorted(unexpected_files)


def iter_authoritative_gt_records(
    gt_root: Any,
    allow_legacy: bool = True,
    gt_partition: Any = "all",
) -> List[Dict[str, Any]]:
    gt_root_p = _resolve_gt_root(gt_root)
    partition_n = _normalize_gt_partition(gt_partition, allow_all=True)
    partitions = list(KNOWN_GT_PARTITIONS) if partition_n == "all" else [partition_n]
    rows: List[Dict[str, Any]] = []
    for partition in partitions:
        part_rows, _unexpected = _scan_partition_inventory(gt_root_p, partition, allow_legacy=allow_legacy)
        rows.extend(part_rows)
    rows.sort(key=lambda row: (str(row.get("gt_partition", "")), _natural_sample_key(str(row.get("sample_id", "")))))
    return rows


def iter_authoritative_gt_sample_ids(
    gt_root: Any,
    allow_legacy: bool = True,
    gt_partition: Any = DEFAULT_GT_PARTITION,
) -> List[str]:
    rows = iter_authoritative_gt_records(gt_root, allow_legacy=allow_legacy, gt_partition=gt_partition)
    sample_ids = {str(row.get("sample_id", "")) for row in rows if str(row.get("sample_id", "")).strip()}
    return sorted(sample_ids, key=_natural_sample_key)


def audit_authoritative_gt_inventory(gt_root: Any, allow_legacy: bool = True) -> Dict[str, Any]:
    gt_root_p = _resolve_gt_root(gt_root)
    rows_bad, unexpected_bad = _scan_partition_inventory(gt_root_p, "bad", allow_legacy=allow_legacy)
    rows_normal, unexpected_normal = _scan_partition_inventory(gt_root_p, "normal", allow_legacy=allow_legacy)
    rows = list(rows_bad) + list(rows_normal)
    rows.sort(key=lambda row: (str(row.get("gt_partition", "")), _natural_sample_key(str(row.get("sample_id", "")))))

    bad_ids = {str(row["sample_id"]) for row in rows_bad if str(row.get("sample_id", "")).strip()}
    normal_ids = {str(row["sample_id"]) for row in rows_normal if str(row.get("sample_id", "")).strip()}
    overlapping_sample_ids = sorted(bad_ids & normal_ids, key=_natural_sample_key)

    issues: List[Dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("pin_exists", False)):
            issues.append({"gt_sample_key": str(row["gt_sample_key"]), "issue": "missing_pin"})
        if not bool(row.get("base_exists", False)):
            issues.append({"gt_sample_key": str(row["gt_sample_key"]), "issue": "missing_base"})

    root_flat_ply_files = sorted(
        [
            str(path.name)
            for path in gt_root_p.iterdir()
            if path.is_file() and str(path.suffix).lower() == ".ply"
        ]
    )
    partition_dirs_present = [name for name in KNOWN_GT_PARTITIONS if (gt_root_p / name).is_dir()]

    return {
        "gt_root": str(gt_root_p),
        "allow_legacy": bool(allow_legacy),
        "partition_dirs_present": list(partition_dirs_present),
        "root_flat_ply_files": list(root_flat_ply_files),
        "sample_count": int(len(rows)),
        "sample_keys": [str(row["gt_sample_key"]) for row in rows],
        "sample_ids_bad": sorted(bad_ids, key=_natural_sample_key),
        "sample_ids_normal": sorted(normal_ids, key=_natural_sample_key),
        "overlapping_sample_ids": list(overlapping_sample_ids),
        "unexpected_files_bad": list(unexpected_bad),
        "unexpected_files_normal": list(unexpected_normal),
        "rows": rows,
        "issues": issues,
        "missing_pin_sample_keys": [str(x["gt_sample_key"]) for x in issues if str(x.get("issue", "")) == "missing_pin"],
        "missing_base_sample_keys": [str(x["gt_sample_key"]) for x in issues if str(x.get("issue", "")) == "missing_base"],
    }
