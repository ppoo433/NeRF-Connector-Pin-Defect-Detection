from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class PlyProperty:
    name: str
    kind: str  # "scalar" | "list"
    dtype: str
    count_dtype: Optional[str] = None  # only for list


@dataclasses.dataclass(frozen=True)
class PlyHeader:
    fmt: str  # ascii | binary_little_endian | binary_big_endian
    version: str
    vertex_count: int
    vertex_properties: Tuple[PlyProperty, ...]
    header_size_bytes: int


_PLY_SCALAR_TYPES_TO_NUMPY = {
    # Per PLY spec variants: char/uchar/short/ushort/int/uint/float/double
    # plus common aliases: int8/uint8/int16/uint16/int32/uint32/float32/float64.
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_NUMPY_TO_PLY_SCALAR_TYPE = {
    np.dtype("i1"): "char",
    np.dtype("u1"): "uchar",
    np.dtype("i2"): "short",
    np.dtype("u2"): "ushort",
    np.dtype("i4"): "int",
    np.dtype("u4"): "uint",
    np.dtype("f4"): "float",
    np.dtype("f8"): "double",
}


def _read_ascii_line(f: BinaryIO) -> bytes:
    line = f.readline()
    if not line:
        raise EOFError("Unexpected EOF while reading PLY header.")
    return line


def _decode_ascii(b: bytes) -> str:
    return b.decode("ascii", errors="ignore").strip()


def parse_ply_header(f: BinaryIO) -> PlyHeader:
    """
    Parse a PLY header from an already-opened binary file handle.

    Supports:
    - format ascii / binary_little_endian / binary_big_endian
    - element vertex <N> with scalar properties only (for vertex reading)
    """
    header_size = 0

    first = _read_ascii_line(f)
    header_size += len(first)
    if _decode_ascii(first) != "ply":
        raise ValueError("Not a PLY file (missing 'ply' magic).")

    fmt = None
    version = None

    vertex_count: Optional[int] = None
    vertex_props: List[PlyProperty] = []
    current_element: Optional[str] = None

    while True:
        raw = _read_ascii_line(f)
        header_size += len(raw)
        line = _decode_ascii(raw)
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        if parts[0] == "comment" or parts[0] == "obj_info":
            continue

        if parts[0] == "format":
            if len(parts) < 3:
                raise ValueError(f"Invalid PLY format line: {line!r}")
            fmt = parts[1]
            version = parts[2]
            continue

        if parts[0] == "element":
            if len(parts) != 3:
                raise ValueError(f"Invalid PLY element line: {line!r}")
            current_element = parts[1]
            if current_element == "vertex":
                try:
                    vertex_count = int(parts[2])
                except Exception as e:
                    raise ValueError(f"Invalid vertex count in PLY header: {line!r}") from e
            continue

        if parts[0] == "property":
            if current_element != "vertex":
                # We only need vertex data for this project; ignore other elements.
                continue
            if len(parts) < 3:
                raise ValueError(f"Invalid PLY property line: {line!r}")
            if parts[1] == "list":
                # list <count_type> <item_type> <name>
                if len(parts) != 5:
                    raise ValueError(f"Invalid PLY list property line: {line!r}")
                vertex_props.append(
                    PlyProperty(
                        name=parts[4],
                        kind="list",
                        dtype=parts[3],
                        count_dtype=parts[2],
                    )
                )
            else:
                # <type> <name>
                if len(parts) != 3:
                    raise ValueError(f"Invalid PLY scalar property line: {line!r}")
                vertex_props.append(PlyProperty(name=parts[2], kind="scalar", dtype=parts[1]))
            continue

        if line in ("end_header", "endheader"):
            break

    if fmt is None or version is None:
        raise ValueError("PLY header missing required 'format' line.")
    if vertex_count is None:
        raise ValueError("PLY header missing required 'element vertex' line.")

    return PlyHeader(
        fmt=fmt,
        version=version,
        vertex_count=vertex_count,
        vertex_properties=tuple(vertex_props),
        header_size_bytes=header_size,
    )


def _numpy_dtype_for_ply_scalar(ply_type: str, *, fmt: str) -> np.dtype:
    base = _PLY_SCALAR_TYPES_TO_NUMPY.get(ply_type)
    if base is None:
        raise ValueError(f"Unsupported PLY scalar type: {ply_type!r}")

    dt = np.dtype(base)
    if dt.itemsize == 1:
        return dt  # endian doesn't matter for 1-byte types

    if fmt == "binary_little_endian":
        return dt.newbyteorder("<")
    if fmt == "binary_big_endian":
        return dt.newbyteorder(">")
    # ascii: the dtype is for parsing only; we don't currently support ascii payload.
    return dt


def read_ply_vertex(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Read vertex element from a PLY file.

    Returns a dict mapping property name -> numpy array of length N.

    Notes:
    - Only supports binary_little_endian / binary_big_endian payloads.
    - Only supports scalar vertex properties (no list properties under vertex).
    """
    path = Path(path)
    with path.open("rb") as f:
        header = parse_ply_header(f)
        if header.fmt not in ("binary_little_endian", "binary_big_endian"):
            raise NotImplementedError(f"Only binary PLY is supported, got: {header.fmt!r}")

        fields: List[Tuple[str, np.dtype]] = []
        for prop in header.vertex_properties:
            if prop.kind != "scalar":
                raise NotImplementedError(
                    f"List vertex property is not supported for reading: {prop.name} ({prop.dtype})"
                )
            fields.append((prop.name, _numpy_dtype_for_ply_scalar(prop.dtype, fmt=header.fmt)))

        dtype = np.dtype(fields)
        arr = np.fromfile(f, dtype=dtype, count=header.vertex_count)

    out: Dict[str, np.ndarray] = {}
    for name in dtype.names or ():
        out[name] = arr[name]
    return out


def write_ply_vertex_binary_little_endian(
    path: str | Path,
    *,
    vertex_props: Dict[str, np.ndarray],
    comments: Optional[Sequence[str]] = None,
) -> None:
    """
    Write a vertex-only PLY with binary_little_endian payload.

    vertex_props:
      - dict of {property_name: array_like}, all arrays must have the same length.
      - Supported dtypes: int8/uint8/int16/uint16/int32/uint32/float32/float64.
    """
    path = Path(path)
    if not vertex_props:
        raise ValueError("vertex_props must not be empty.")

    # Preserve insertion order (Python 3.7+ guarantees dict order).
    prop_names = list(vertex_props.keys())
    n = None
    for k in prop_names:
        a = np.asarray(vertex_props[k])
        if a.ndim != 1:
            raise ValueError(f"vertex_props[{k!r}] must be a 1D array, got shape={a.shape}")
        if n is None:
            n = int(a.shape[0])
        elif int(a.shape[0]) != n:
            raise ValueError(f"All vertex property arrays must have same length, {k!r} differs.")
    assert n is not None

    # Build dtype (force little-endian for multi-byte scalars).
    fields: List[Tuple[str, np.dtype]] = []
    header_prop_lines: List[str] = []
    for name in prop_names:
        arr = np.asarray(vertex_props[name])
        dt = arr.dtype
        # Normalize endianness and dtype kind.
        dt0 = np.dtype(dt).newbyteorder("=")  # native
        dt_base = np.dtype(dt0.str.replace("<", "").replace(">", "").replace("=", ""))
        ply_t = _NUMPY_TO_PLY_SCALAR_TYPE.get(dt_base)
        if ply_t is None:
            raise ValueError(f"Unsupported dtype for PLY write: {name} dtype={dt} (base={dt_base})")

        dt_le = dt_base if dt_base.itemsize == 1 else dt_base.newbyteorder("<")
        fields.append((name, dt_le))
        header_prop_lines.append(f"property {ply_t} {name}")

    dtype = np.dtype(fields)

    structured = np.empty(n, dtype=dtype)
    for name in prop_names:
        structured[name] = np.asarray(vertex_props[name], dtype=dtype.fields[name][0], order="C")

    lines: List[str] = []
    lines.append("ply")
    lines.append("format binary_little_endian 1.0")
    if comments:
        for c in comments:
            # PLY comments are ASCII; strip newlines and non-ASCII best-effort.
            c_safe = "".join(ch for ch in str(c).replace("\n", " ").replace("\r", " ") if ord(ch) < 128)
            lines.append(f"comment {c_safe}")
    lines.append(f"element vertex {n}")
    lines.extend(header_prop_lines)
    lines.append("end_header")
    header_bytes = ("\n".join(lines) + "\n").encode("ascii", errors="ignore")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header_bytes)
        structured.tofile(f)

