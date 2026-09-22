"""Download and arrange the six datasets used by LADDER.

The training code expects datasets next to ``src``::

    <project>/src/
    <project>/datasets/CWRU_10/
    <project>/datasets/XJTU/
    <project>/datasets/SEU/
    <project>/datasets/MFPT/
    <project>/datasets/PU/
    <project>/datasets/IMS/

Downloads only start when a selected dataset is missing.  Set
``LADDER_AUTO_DOWNLOAD=0`` to disable that behaviour.  This module can also be
run directly from the ``src`` directory::

    python -m utils.download_datasets --datasets CWRU_10 SEU MFPT

Large archives are kept under ``datasets/.downloads`` only while they are
needed.  Set ``LADDER_KEEP_DATASET_ARCHIVES=1`` to retain them.

For CWRU, the default ``legacy`` variant reproduces the historical LADDER
``1750-0.000-Normal.mat`` file byte-for-byte.  Set
``LADDER_CWRU_VARIANT=official`` (or use ``--cwru-variant official`` when this
module is run directly) to retain CWRU's unmodified official ``99.mat`` file.

The module downloads data from its providers; it does not grant additional
rights to the datasets.  Review and follow each provider's terms and citation
requirements before use or redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


PathLike = Union[os.PathLike, str]


SUPPORTED_DATASETS: Tuple[str, ...] = (
    "CWRU_10",
    "XJTU",
    "SEU",
    "MFPT",
    "PU",
    "IMS",
)

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_USER_AGENT = "LADDER-dataset-downloader/1.0"
_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_RETRIES = 5
_DOWNLOAD_TIMEOUT = 120
_RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded or arranged safely."""


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise DatasetDownloadError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {value!r}."
    )


def _canonical_dataset_name(dataset_name: str) -> str:
    aliases = {
        "CWRU": "CWRU_10",
        "CWRU10": "CWRU_10",
        "CWRU_10": "CWRU_10",
        "XJTU": "XJTU",
        "XJTU-SY": "XJTU",
        "SEU": "SEU",
        "MFPT": "MFPT",
        "PU": "PU",
        "PADERBORN": "PU",
        "IMS": "IMS",
    }
    key = dataset_name.strip().upper()
    try:
        return aliases[key]
    except KeyError as exc:
        raise DatasetDownloadError(
            f"Unsupported dataset {dataset_name!r}. Choose from {SUPPORTED_DATASETS}."
        ) from exc


def _project_root(src_dir: PathLike) -> Path:
    src_path = Path(src_dir).expanduser().resolve()
    if src_path.name != "src":
        raise DatasetDownloadError(
            f"Expected src_dir to point to the project's 'src' directory, got: {src_path}"
        )
    return src_path.parent


def dataset_path(dataset_name: str, src_dir: PathLike) -> Path:
    """Return the path already used by ``utils.constants`` for a dataset."""

    name = _canonical_dataset_name(dataset_name)
    return _project_root(src_dir) / "datasets" / name


def _regular_files(path: Path, suffix: Optional[str] = None) -> List[Path]:
    if not path.is_dir():
        return []
    files = [item for item in path.iterdir() if item.is_file()]
    if suffix is not None:
        files = [item for item in files if item.suffix.lower() == suffix.lower()]
    return files


_SEU_FILES: Dict[str, Tuple[str, ...]] = {
    "bearingset": (
        "health_20_0.csv",
        "health_30_2.csv",
        "ball_20_0.csv",
        "ball_30_2.csv",
        "comb_20_0.csv",
        "comb_30_2.csv",
        "inner_20_0.csv",
        "inner_30_2.csv",
        "outer_20_0.csv",
        "outer_30_2.csv",
    ),
    "gearset": (
        "Health_20_0.csv",
        "Health_30_2.csv",
        "Chipped_20_0.csv",
        "Chipped_30_2.csv",
        "Miss_20_0.csv",
        "Miss_30_2.csv",
        "Root_20_0.csv",
        "Root_30_2.csv",
        "Surface_20_0.csv",
        "Surface_30_2.csv",
    ),
}

_MFPT_FILES: Tuple[str, ...] = (
    "baseline_1.mat",
    "baseline_2.mat",
    "baseline_3.mat",
    "OuterRaceFault_1.mat",
    "OuterRaceFault_2.mat",
    "OuterRaceFault_3.mat",
    "InnerRaceFault_vload_1.mat",
    "InnerRaceFault_vload_2.mat",
    "InnerRaceFault_vload_3.mat",
    "InnerRaceFault_vload_4.mat",
    "InnerRaceFault_vload_5.mat",
    "InnerRaceFault_vload_6.mat",
    "InnerRaceFault_vload_7.mat",
)

# All 32 official Paderborn bearing experiments are downloaded so the local
# PU directory mirrors the complete provider dataset.  This list controls only
# dataset acquisition/readiness; the existing training ACT_LABELS and PU
# preprocessing code continue to select the same experimental subset as before.
_PU_LABELS: Tuple[str, ...] = (
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006",
    "KA01",
    "KA03",
    "KA04",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KB23",
    "KB24",
    "KB27",
    "KI01",
    "KI03",
    "KI04",
    "KI05",
    "KI07",
    "KI08",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
)

_IMS_TIMESTAMP = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")

_IMS_EXPECTED_COUNTS: Dict[str, int] = {
    "1st_test": 2156,
    "2nd_test": 984,
    "3rd_test": 6324,
}

_CWRU_VARIANT_ENV = "LADDER_CWRU_VARIANT"
_CWRU_VARIANTS: Tuple[str, str] = ("legacy", "official")
_CWRU_SPECIAL_FILENAME = "1750-0.000-Normal.mat"
_CWRU_LEGACY_SPECIAL_SIZE = 3_896_927
_CWRU_LEGACY_SPECIAL_SHA256 = (
    "c5b8dfd42c408e6fd3bce4b7ed3754d1cbd36ca72f4e6c1ae365169ca7a4a41a"
)
_CWRU_OFFICIAL_SPECIAL_SIZE = 15_503_928
_CWRU_OFFICIAL_SPECIAL_SHA256 = (
    "4b97e6b5361f45efb6951dc3b1aebcdb3b89cb69d0f96d6f5c297dd9f45eee75"
)


def _cwru_variant() -> str:
    """Return the requested CWRU layout, defaulting to LADDER's legacy copy."""

    value = os.getenv(_CWRU_VARIANT_ENV, "legacy").strip().lower()
    if value not in _CWRU_VARIANTS:
        raise DatasetDownloadError(
            f"{_CWRU_VARIANT_ENV} must be one of {_CWRU_VARIANTS}, got {value!r}."
        )
    return value


def _cwru_ready(path: Path) -> bool:
    if len(_regular_files(path, ".mat")) != 40:
        return False

    special_file = path / _CWRU_SPECIAL_FILENAME
    if not special_file.is_file():
        return False

    if _cwru_variant() == "legacy":
        expected_size = _CWRU_LEGACY_SPECIAL_SIZE
        expected_sha256 = _CWRU_LEGACY_SPECIAL_SHA256
    else:
        expected_size = _CWRU_OFFICIAL_SPECIAL_SIZE
        expected_sha256 = _CWRU_OFFICIAL_SPECIAL_SHA256

    return (
        special_file.stat().st_size == expected_size
        and _sha256_file(special_file) == expected_sha256
    )


def _xjtu_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    expected = (("Bearing1_", 5), ("Bearing2_", 5), ("Bearing3_", 5))
    condition_dirs = sorted(item for item in path.iterdir() if item.is_dir())
    if len(condition_dirs) != 3:
        return False
    for condition, (prefix, count) in zip(condition_dirs, expected):
        bearings = sorted(
            item for item in condition.iterdir()
            if item.is_dir() and item.name.startswith(prefix)
        )
        if len(bearings) != count:
            return False
        if any(len(list(bearing.glob("*.csv"))) < 4 for bearing in bearings):
            return False
    return True


def _seu_ready(path: Path) -> bool:
    return all(
        (path / folder / filename).is_file()
        for folder, filenames in _SEU_FILES.items()
        for filename in filenames
    )


def _mfpt_ready(path: Path) -> bool:
    return all((path / filename).is_file() for filename in _MFPT_FILES)


def _pu_label_ready(path: Path, label: str) -> bool:
    # Each official archive contains 20 measurements for each of four operating
    # settings, i.e. 80 MAT files for one bearing state.
    return (path / label).is_dir() and len(list((path / label).glob("*.mat"))) >= 80


def _pu_ready(path: Path) -> bool:
    return all(_pu_label_ready(path, label) for label in _PU_LABELS)


def _ims_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    for folder, expected_count in _IMS_EXPECTED_COUNTS.items():
        directory = path / folder
        if not directory.is_dir():
            return False
        count = sum(
            1 for item in directory.iterdir()
            if item.is_file() and _IMS_TIMESTAMP.fullmatch(item.name)
        )
        if count != expected_count:
            return False
    return True


_READY_CHECKS: Dict[str, Callable[[Path], bool]] = {
    "CWRU_10": _cwru_ready,
    "XJTU": _xjtu_ready,
    "SEU": _seu_ready,
    "MFPT": _mfpt_ready,
    "PU": _pu_ready,
    "IMS": _ims_ready,
}


def dataset_is_ready(dataset_name: str, src_dir: PathLike) -> bool:
    """Return whether the on-disk layout satisfies the current loader."""

    name = _canonical_dataset_name(dataset_name)
    return _READY_CHECKS[name](dataset_path(name, src_dir))


def _print_progress(filename: str, received: int, total: int, last_percent: int) -> int:
    if total <= 0:
        if received == 0 or received // (100 * _CHUNK_SIZE) > last_percent:
            downloaded_mb = received / (1024 * 1024)
            print(f"  {filename}: {downloaded_mb:.0f} MiB downloaded", flush=True)
            return received // (100 * _CHUNK_SIZE)
        return last_percent
    percent = min(100, int(received * 100 / total))
    if percent == 100 or percent >= last_percent + 5:
        print(f"  {filename}: {percent}%", flush=True)
        return percent
    return last_percent


def _download_url(url: str, destination: Path) -> Path:
    """Download a URL atomically with retry and ``.part`` resume support.

    A failed transfer keeps the partial file.  The next retry (or a later
    program run) requests the remaining bytes with an HTTP Range header.
    If the server ignores Range, the partial file is safely restarted rather
    than appended to a full response.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached archive: {destination}")
        return destination

    partial = destination.with_name(destination.name + ".part")
    last_error: Optional[BaseException] = None

    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        resume_at = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": _USER_AGENT}
        if resume_at:
            headers["Range"] = f"bytes={resume_at}-"

        request = urllib.request.Request(url, headers=headers)
        action = "Resuming" if resume_at else "Downloading"
        print(
            f"{action} {url} "
            f"(attempt {attempt}/{_DOWNLOAD_RETRIES}, "
            f"partial={resume_at / (1024 * 1024):.2f} MiB)"
        )

        try:
            with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
                status = getattr(response, "status", response.getcode())
                append = resume_at > 0 and status == 206

                if append:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.search(r"/(\d+)$", content_range)
                    total = int(match.group(1)) if match else (
                        resume_at
                        + int(response.headers.get("Content-Length", "0") or 0)
                    )
                    received = resume_at
                    mode = "ab"
                else:
                    # Some servers ignore Range requests and return HTTP 200.
                    # Restart this file so a complete response is never appended
                    # to an existing partial file.
                    if resume_at:
                        print(
                            f"  {destination.name}: server did not honor Range; "
                            "restarting this file from byte 0."
                        )
                    total = int(response.headers.get("Content-Length", "0") or 0)
                    received = 0
                    mode = "wb"

                last_percent = -5
                with partial.open(mode) as output:
                    while True:
                        block = response.read(_CHUNK_SIZE)
                        if not block:
                            break
                        output.write(block)
                        output.flush()
                        received += len(block)
                        last_percent = _print_progress(
                            destination.name, received, total, last_percent
                        )

            if not partial.is_file() or partial.stat().st_size == 0:
                raise OSError(f"Downloaded an empty file from {url}")

            # urllib may occasionally return EOF without raising even though
            # Content-Length/Content-Range says more bytes were expected.
            # Treat that as an interrupted transfer so the next attempt resumes.
            if total > 0 and received < total:
                raise OSError(
                    f"Connection ended early: received {received} of {total} bytes"
                )

            os.replace(partial, destination)
            return destination

        except urllib.error.HTTPError as exc:
            # HTTP 416 can mean our retained .part already contains the complete
            # object.  Accept it only when the server reports the same size.
            if exc.code == 416 and partial.is_file():
                content_range = exc.headers.get("Content-Range", "") if exc.headers else ""
                match = re.search(r"\*/(\d+)$", content_range)
                if match and partial.stat().st_size == int(match.group(1)):
                    os.replace(partial, destination)
                    return destination

            last_error = exc
            retryable = exc.code in _RETRYABLE_HTTP_CODES
            if not retryable or attempt == _DOWNLOAD_RETRIES:
                raise DatasetDownloadError(
                    f"Failed to download {url}: {exc}. "
                    f"Partial data was kept at {partial}."
                ) from exc

        except (OSError, urllib.error.URLError) as exc:
            # SSL EOF errors, TLS disconnects, socket resets, timeouts and
            # related urllib failures land here.  Keep .part and resume.
            last_error = exc
            if attempt == _DOWNLOAD_RETRIES:
                raise DatasetDownloadError(
                    f"Failed to download {url} after {_DOWNLOAD_RETRIES} attempts: {exc}. "
                    f"Partial data was kept at {partial}."
                ) from exc

        wait_seconds = min(2 ** attempt, 30)
        print(
            f"  {destination.name}: transient download error: {last_error}. "
            f"Retrying in {wait_seconds} seconds...",
            flush=True,
        )
        time.sleep(wait_seconds)

    # The loop either returns or raises.  This is only a defensive fallback.
    raise DatasetDownloadError(
        f"Failed to download {url}. Partial data was kept at {partial}."
    )


def _validate_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    member = PurePosixPath(info.filename.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts:
        raise DatasetDownloadError(f"Unsafe path in ZIP archive: {info.filename!r}")
    unix_mode = (info.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise DatasetDownloadError(f"Symbolic link rejected in ZIP archive: {info.filename!r}")
    return member


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    if not zipfile.is_zipfile(archive):
        raise DatasetDownloadError(f"The downloaded file is not a valid ZIP archive: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            member = _validate_zip_member(info)
            if not member.parts:
                continue
            target = destination.joinpath(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=_CHUNK_SIZE)


def _install_directory(staged: Path, target: Path) -> None:
    """Install a fully validated staged directory without deleting user data."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_dir():
            raise DatasetDownloadError(
                f"Refusing to overwrite non-directory dataset target: {target}"
            )
        if any(target.iterdir()):
            raise DatasetDownloadError(
                f"Refusing to overwrite the incomplete non-empty directory {target}. "
                "Move it aside, complete it manually, or choose a clean project directory."
            )
        target.rmdir()
    os.replace(staged, target)


def _require_empty_install_target(target: Path) -> None:
    """Fail early rather than downloading into an ambiguous existing layout."""

    if not target.exists():
        return
    if not target.is_dir() or any(target.iterdir()):
        raise DatasetDownloadError(
            f"Dataset directory {target} exists but is incomplete. Refusing to overwrite it; "
            "move it aside or complete it manually before retrying."
        )


def _cleanup_archive(archive: Path, keep_archive: bool) -> None:
    if not keep_archive and archive.exists():
        archive.unlink()


_CWRU_CONDITIONS: Tuple[Tuple[str, Tuple[int, int, int, int]], ...] = (
    ("0.000-Normal", (97, 98, 99, 100)),
    ("0.007-Ball", (118, 119, 120, 121)),
    ("0.007-InnerRace", (105, 106, 107, 108)),
    ("0.007-OuterRace6", (130, 131, 132, 133)),
    ("0.014-Ball", (185, 186, 187, 188)),
    ("0.014-InnerRace", (169, 170, 171, 172)),
    ("0.014-OuterRace6", (197, 198, 199, 200)),
    ("0.021-Ball", (222, 223, 224, 225)),
    ("0.021-InnerRace", (209, 210, 211, 212)),
    ("0.021-OuterRace6", (234, 235, 236, 237)),
)

# Preserve the exact filename convention used by the original LADDER CWRU_10
# directory.  The four entries correspond to the official CWRU 0, 1, 2 and
# 3 hp files respectively; their nominal shaft speeds are 1797, 1772, 1750
# and 1730 rpm.  The legacy CWRU preprocessor sorts these filenames
# lexicographically, so retaining this convention is part of reproducing the
# original experiment rather than merely a cosmetic rename.
_CWRU_RPMS: Tuple[str, str, str, str] = ("1797", "1772", "1750", "1730")

# CWRU's official 99.mat contains both X098 and X099.  The historical LADDER
# file named 1750-0.000-Normal.mat retained only the X098 pair and was written
# by MATLAB using zlib level 6 with a full flush every 8192 bytes.  The helper
# below deliberately recreates that historical file exactly for reproducibility.
# This is a compatibility choice, not a correction of the CWRU metadata: X099
# is the signal that semantically belongs to the 1750-rpm condition.
_CWRU_LEGACY_MATRIX_NAMES: Tuple[str, str] = (
    "X098_DE_time",
    "X098_FE_time",
)
_CWRU_LEGACY_MAT_HEADER = (
    b"MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: "
    b"Sun May 19 01:49:09 2019"
).ljust(116, b" ") + (b"\x00" * 8) + b"\x00\x01IM"
_MAT5_MI_MATRIX = 14
_MAT5_MI_COMPRESSED = 15
_MATLAB_ZLIB_CHUNK_SIZE = 8192


def _extract_cwru_legacy_matrices(source: Path) -> Dict[str, bytes]:
    """Extract X098 MAT-v5 matrix elements from the official CWRU 99.mat."""

    content = source.read_bytes()
    if len(content) < 128 or not content.startswith(b"MATLAB 5.0 MAT-file"):
        raise DatasetDownloadError(f"CWRU source is not a MATLAB v5 file: {source}")

    endian_indicator = content[126:128]
    if endian_indicator == b"IM":
        byte_order = "<"
    elif endian_indicator == b"MI":
        byte_order = ">"
    else:
        raise DatasetDownloadError(
            f"CWRU source has an invalid MATLAB endian marker: {source}"
        )

    selected: Dict[str, bytes] = {}
    offset = 128
    while offset + 8 <= len(content):
        data_type, payload_size = struct.unpack_from(byte_order + "II", content, offset)
        element_end = offset + 8 + payload_size
        if element_end > len(content):
            raise DatasetDownloadError(f"CWRU MAT element is truncated: {source}")

        if data_type == _MAT5_MI_MATRIX:
            matrix_element = content[offset:element_end]
        elif data_type == _MAT5_MI_COMPRESSED:
            try:
                matrix_element = zlib.decompress(content[offset + 8:element_end])
            except zlib.error as exc:
                raise DatasetDownloadError(
                    f"Cannot decompress a CWRU MAT element in {source}."
                ) from exc
        else:
            matrix_element = b""

        # Array names occur in the small metadata prefix of each miMATRIX.
        metadata_prefix = matrix_element[:256]
        for name in _CWRU_LEGACY_MATRIX_NAMES:
            if name.encode("ascii") in metadata_prefix:
                if name in selected:
                    raise DatasetDownloadError(
                        f"CWRU source contains duplicate matrix {name}: {source}"
                    )
                selected[name] = matrix_element

        # MATLAB/Scipy miCOMPRESSED elements are not padded; ordinary MAT-v5
        # elements are aligned to an eight-byte boundary.
        if data_type == _MAT5_MI_COMPRESSED:
            offset = element_end
        else:
            offset = (element_end + 7) & ~7

    missing = [name for name in _CWRU_LEGACY_MATRIX_NAMES if name not in selected]
    if missing:
        raise DatasetDownloadError(
            f"CWRU source {source} is missing legacy matrices: {', '.join(missing)}"
        )
    return selected


def _matlab_legacy_compress(data: bytes) -> bytes:
    """Reproduce the chunked zlib stream written by the legacy MATLAB file."""

    compressor = zlib.compressobj(
        level=6,
        method=zlib.DEFLATED,
        wbits=zlib.MAX_WBITS,
        memLevel=8,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )
    pieces: List[bytes] = []
    for start in range(0, len(data), _MATLAB_ZLIB_CHUNK_SIZE):
        end = min(start + _MATLAB_ZLIB_CHUNK_SIZE, len(data))
        pieces.append(compressor.compress(data[start:end]))
        if end < len(data):
            pieces.append(compressor.flush(zlib.Z_FULL_FLUSH))
    pieces.append(compressor.flush(zlib.Z_FINISH))
    return b"".join(pieces)


def _write_cwru_legacy_normal(source: Path, destination: Path) -> None:
    """Create the historical LADDER 1750 normal MAT file byte-for-byte."""

    matrices = _extract_cwru_legacy_matrices(source)
    output = bytearray(_CWRU_LEGACY_MAT_HEADER)
    for name in _CWRU_LEGACY_MATRIX_NAMES:
        compressed = _matlab_legacy_compress(matrices[name])
        output.extend(struct.pack("<II", _MAT5_MI_COMPRESSED, len(compressed)))
        output.extend(compressed)

    partial = destination.with_name(destination.name + ".part")
    try:
        partial.write_bytes(output)
        if (
            partial.stat().st_size != _CWRU_LEGACY_SPECIAL_SIZE
            or _sha256_file(partial) != _CWRU_LEGACY_SPECIAL_SHA256
        ):
            raise DatasetDownloadError(
                "Generated CWRU legacy 1750 normal file failed exact verification."
            )
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def _download_cwru(target: Path, cache: Path, keep_archive: bool) -> None:
    _require_empty_install_target(target)
    variant = _cwru_variant()
    print(f"Preparing CWRU_10 variant: {variant}")
    source_cache = cache / "CWRU_10"
    source_cache.mkdir(parents=True, exist_ok=True)
    cached_files: List[Path] = []
    with tempfile.TemporaryDirectory(prefix=".CWRU_10-", dir=str(target.parent)) as temp_name:
        staged = Path(temp_name) / "CWRU_10"
        staged.mkdir()

        # Keep the original LADDER naming exactly: e.g.
        #   1730-0.000-Normal.mat
        #   1730-0.007-Ball.mat
        #   ...
        #   1797-0.021-OuterRace6.mat
        # The source file IDs remain the same official 12-kHz drive-end files.
        for load_index, rpm in enumerate(_CWRU_RPMS):
            for label, file_ids in _CWRU_CONDITIONS:
                file_id = file_ids[load_index]
                cached_file = _download_url(
                    f"https://engineering.case.edu/sites/default/files/{file_id}.mat",
                    source_cache / f"{file_id}.mat",
                )
                cached_files.append(cached_file)
                output_file = staged / f"{rpm}-{label}.mat"
                if variant == "legacy" and file_id == 99:
                    _write_cwru_legacy_normal(cached_file, output_file)
                else:
                    shutil.copy2(cached_file, output_file)

        if not _cwru_ready(staged):
            raise DatasetDownloadError(
                f"CWRU {variant} download did not produce the expected verified 40 MAT files."
            )
        _install_directory(staged, target)

    if not keep_archive:
        for cached_file in cached_files:
            _cleanup_archive(cached_file, keep_archive=False)


def _find_xjtu_conditions(extracted: Path) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    bearing_pattern = re.compile(r"^Bearing([123])_[1-5]$")
    for directory in extracted.rglob("*"):
        if not directory.is_dir():
            continue
        matches = [
            item for item in directory.iterdir()
            if item.is_dir() and bearing_pattern.fullmatch(item.name)
        ]
        if len(matches) != 5:
            continue
        condition_numbers = {int(bearing_pattern.fullmatch(item.name).group(1)) for item in matches}
        if len(condition_numbers) == 1:
            condition_number = condition_numbers.pop()
            if condition_number in found:
                raise DatasetDownloadError(
                    f"Found more than one candidate for XJTU condition {condition_number}."
                )
            found[condition_number] = directory
    if set(found) != {1, 2, 3}:
        raise DatasetDownloadError(
            "Could not find the three XJTU condition folders containing Bearing1_1 ... Bearing3_5."
        )
    return found


# filename, Google Drive file id, expected byte size, SHA-256.
# The size/hash metadata corresponds to the publicly mirrored original XJTU-SY
# six-volume RAR release.  Validating the cache is important because an interrupted
# Google Drive download can leave a non-empty but unusable volume behind.
_XJTU_RAR_PARTS: Tuple[Tuple[str, str, int, str], ...] = (
    (
        "XJTU-SY_Bearing_Datasets.part01.rar",
        "1ATvZuD6j3bPxhyR07Zm-PURmOC4b4uRn",
        744_488_960,
        "c500657353f089a4ab50212ff4ddfc7b982729e92e2b127440c8ad881d80b968",
    ),
    (
        "XJTU-SY_Bearing_Datasets.part02.rar",
        "162KvWNIpBGtd7EDWo4yP1j5XsaoNHOYU",
        744_488_960,
        "4dfa6286a8e9c7cec1e925b347642349f36e27ab3f703c90d1d08977b7b4f61f",
    ),
    (
        "XJTU-SY_Bearing_Datasets.part03.rar",
        "1NvzrGW-KOSy48OZmiFxlE3TPV4CKAcw0",
        744_488_960,
        "6929f531284f79e0209246b4ee23b23dd8d4faac1c0c3ff8fb131cda4a18bd3a",
    ),
    (
        "XJTU-SY_Bearing_Datasets.part04.rar",
        "1VuQ5-mK11p1S2pTxUZaH_IxOwUlsmN0S",
        744_488_960,
        "2245825acd29bed3b95e7a77879a3fbadf20f4ad4f99486384c714174621597d",
    ),
    (
        "XJTU-SY_Bearing_Datasets.part05.rar",
        "1WH4OU4MLaMGQkbh6DghxPA5Dwvsq8tEf",
        744_488_960,
        "e1b0a41a32b865e48ea7981c56f952a960d94335f3496b72eb4a65932f1671bd",
    ),
    (
        "XJTU-SY_Bearing_Datasets.part06.rar",
        "1wzQzQUx6-J8DuGczT81OkrkTgOUwL-I_",
        722_155_640,
        "df1854821a9d481104476379f7bc045ea1e101427c6e36145cd7677dc7c4a684",
    ),
)

_XJTU_PART_RETRIES = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(_CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _xjtu_rar_part_is_valid(
    path: Path, expected_size: int, expected_sha256: str, *, announce: bool = True
) -> bool:
    """Verify one cached XJTU multi-volume RAR part before it is reused."""

    if not path.is_file():
        return False

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        if announce:
            print(
                f"Cached XJTU part is incomplete: {path.name} "
                f"({actual_size:,} bytes, expected {expected_size:,}); it will be re-downloaded."
            )
        return False

    if announce:
        print(f"Verifying cached XJTU archive part: {path.name}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        if announce:
            print(
                f"Cached XJTU part failed SHA-256 verification: {path.name}; "
                "it will be re-downloaded."
            )
        return False
    return True


def _download_xjtu_rar_parts(destination: Path) -> List[Path]:
    """Download and verify the six XJTU-SY multi-volume RAR files."""

    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise DatasetDownloadError(
            "XJTU's author distributes the data as a multi-part RAR set on Google Drive. "
            "Install the small download helper first with 'pip install gdown', or set "
            "LADDER_XJTU_URL to a direct ZIP mirror supplied by the dataset author."
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    parts: List[Path] = []

    for filename, file_id, expected_size, expected_sha256 in _XJTU_RAR_PARTS:
        output = destination / filename

        if _xjtu_rar_part_is_valid(output, expected_size, expected_sha256):
            print(f"Using verified cached archive part: {output}")
            parts.append(output)
            continue

        # A stale/truncated file must not be handed to gdown as though it were
        # complete.  Remove only this bad volume; verified volumes stay cached.
        if output.exists():
            output.unlink()

        last_error: Optional[BaseException] = None
        downloaded = False
        for attempt in range(1, _XJTU_PART_RETRIES + 1):
            print(
                f"Downloading XJTU-SY archive part: {filename} "
                f"(attempt {attempt}/{_XJTU_PART_RETRIES})"
            )
            try:
                result = gdown.download(
                    id=file_id,
                    output=str(output),
                    quiet=False,
                    resume=True,
                )
                if not result or not output.is_file():
                    raise DatasetDownloadError(
                        f"Google Drive did not return XJTU archive part {filename}."
                    )

                if _xjtu_rar_part_is_valid(
                    output, expected_size, expected_sha256, announce=False
                ):
                    print(f"Verified XJTU archive part: {filename}")
                    downloaded = True
                    break

                last_error = DatasetDownloadError(
                    f"Downloaded XJTU archive part {filename} failed size/SHA-256 verification."
                )
            except Exception as exc:
                last_error = exc

            # Do not resume from bytes that have already failed integrity checks.
            if output.exists():
                output.unlink()
            if attempt < _XJTU_PART_RETRIES:
                delay = min(2 ** attempt, 10)
                print(f"Retrying {filename} in {delay} seconds...")
                time.sleep(delay)

        if not downloaded:
            raise DatasetDownloadError(
                f"Failed to obtain a valid XJTU archive part {filename} after "
                f"{_XJTU_PART_RETRIES} attempts. The bad cache file was removed. "
                "Check the Google Drive connection or set LADDER_XJTU_URL to a direct "
                "ZIP mirror supplied by the dataset author."
            ) from last_error

        parts.append(output)

    return parts


def _download_xjtu(target: Path, cache: Path, keep_archive: bool) -> None:
    _require_empty_install_target(target)
    override = os.getenv("LADDER_XJTU_URL")
    archives: List[Path]
    if override:
        archive = _download_url(override, cache / "XJTU-SY_Bearing_Datasets.zip")
        if not zipfile.is_zipfile(archive):
            raise DatasetDownloadError(
                "LADDER_XJTU_URL must point to a ZIP containing the three condition folders."
            )
        archives = [archive]
    else:
        # The official Google Drive folder contains six multi-part RAR files.
        # Check extraction support before starting this very large download.
        _rar_command(
            cache / "XJTU" / _XJTU_RAR_PARTS[0][0],
            cache / "XJTU" / "probe",
            "XJTU",
        )
        print(
            "XJTU notice: the author's dataset is a large six-part RAR download; "
            "allow ample disk space for both the archives and extracted CSV files."
        )
        archives = _download_xjtu_rar_parts(cache / "XJTU")

    with tempfile.TemporaryDirectory(prefix=".XJTU-", dir=str(target.parent)) as temp_name:
        temp = Path(temp_name)
        extracted = temp / "extracted"
        extracted.mkdir()
        if override:
            _extract_zip_safely(archives[0], extracted)
        else:
            command = _rar_command(archives[0], extracted, "XJTU")
            print(f"Extracting XJTU-SY with {Path(command[0]).name}")
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise DatasetDownloadError(f"Failed to extract XJTU-SY: {exc}") from exc
        conditions = _find_xjtu_conditions(extracted)
        staged = temp / "XJTU"
        staged.mkdir()
        output_names = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}
        for number, source in conditions.items():
            shutil.copytree(source, staged / output_names[number])
        if not _xjtu_ready(staged):
            raise DatasetDownloadError("XJTU archive layout does not satisfy the current loader.")
        _install_directory(staged, target)
    for archive in archives:
        _cleanup_archive(archive, keep_archive)


def _download_seu(target: Path, cache: Path, keep_archive: bool) -> None:
    _require_empty_install_target(target)
    commit = "9732a2820f4cdf91fa7d1d0cc95442376660f5dd"
    source_cache = cache / "SEU"
    base_url = os.getenv(
        "LADDER_SEU_BASE_URL",
        f"https://raw.githubusercontent.com/cathysiyu/Mechanical-datasets/{commit}/gearbox",
    ).rstrip("/")
    cached_files: List[Path] = []
    with tempfile.TemporaryDirectory(prefix=".SEU-", dir=str(target.parent)) as temp_name:
        temp = Path(temp_name)
        staged = temp / "SEU"
        staged.mkdir()
        for folder, filenames in _SEU_FILES.items():
            output_folder = staged / folder
            output_folder.mkdir()
            for filename in filenames:
                source_file = _download_url(
                    f"{base_url}/{folder}/{filename}",
                    source_cache / folder / filename,
                )
                cached_files.append(source_file)
                shutil.copy2(source_file, output_folder / filename)
        if not _seu_ready(staged):
            raise DatasetDownloadError("SEU download is missing files required by preprocessing.")
        _install_directory(staged, target)
    if not keep_archive:
        for cached_file in cached_files:
            _cleanup_archive(cached_file, keep_archive=False)


def _download_mfpt(target: Path, cache: Path, keep_archive: bool) -> None:
    _require_empty_install_target(target)
    commit = "d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d"
    archive = _download_url(
        os.getenv(
            "LADDER_MFPT_URL",
            f"https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/archive/{commit}.zip",
        ),
        cache / f"MFPT-RollingElementBearingFaultDiagnosis-Data-{commit}.zip",
    )
    with tempfile.TemporaryDirectory(prefix=".MFPT-", dir=str(target.parent)) as temp_name:
        temp = Path(temp_name)
        extracted = temp / "extracted"
        _extract_zip_safely(archive, extracted)
        staged = temp / "MFPT"
        staged.mkdir()
        for filename in _MFPT_FILES:
            candidates = list(extracted.rglob(filename))
            if len(candidates) != 1:
                raise DatasetDownloadError(
                    f"Expected exactly one MFPT file named {filename}, found {len(candidates)}."
                )
            shutil.copy2(candidates[0], staged / filename)
        if not _mfpt_ready(staged):
            raise DatasetDownloadError("MFPT archive is missing files required by preprocessing.")
        _install_directory(staged, target)
    _cleanup_archive(archive, keep_archive)


def _rar_command(archive: Path, destination: Path, dataset_name: str) -> List[str]:
    """Return a command for extracting a RAR archive.

    ``dataset_name`` is used only to produce an accurate error message when no
    supported extractor is installed.  Both XJTU and PU use this helper.
    """

    seven_zip = shutil.which("7zz") or shutil.which("7z") or shutil.which("7z.exe")
    if seven_zip:
        return [seven_zip, "x", "-y", f"-o{destination}", str(archive)]
    unar = shutil.which("unar")
    if unar:
        return [unar, "-f", "-o", str(destination), str(archive)]
    unrar = shutil.which("unrar") or shutil.which("unrar.exe")
    if unrar:
        return [unrar, "x", "-o+", "-idq", str(archive), str(destination) + os.sep]
    bsdtar = shutil.which("bsdtar")
    if bsdtar:
        return [bsdtar, "-xf", str(archive), "-C", str(destination)]
    raise DatasetDownloadError(
        f"{dataset_name} is distributed as RAR archives, but no supported "
        "RAR extractor was found. Install one of 7-Zip/7zz, unar, unrar, "
        "or bsdtar and run the command again."
    )


def _download_pu(target: Path, cache: Path, keep_archive: bool) -> None:
    # Fail before downloading a multi-hundred-megabyte archive if this machine
    # cannot extract the provider's RAR format.
    _rar_command(cache / "PU" / "probe.rar", cache / "PU" / "probe", "PU")
    print(
        "PU notice: downloading the complete 32-bearing official dataset requires "
        "several GB of compressed archives; allow ample disk space and cite the data provider."
    )
    target.mkdir(parents=True, exist_ok=True)
    unexpected = [item.name for item in target.iterdir() if item.name not in _PU_LABELS]
    if unexpected:
        raise DatasetDownloadError(
            f"Unexpected entries in {target}: {unexpected}. Refusing to mix layouts."
        )

    for label in _PU_LABELS:
        if _pu_label_ready(target, label):
            print(f"PU {label}: already present")
            continue
        label_target = target / label
        if label_target.exists():
            raise DatasetDownloadError(
                f"PU directory {label_target} is incomplete. Move it aside before retrying."
            )

        archive = _download_url(
            f"https://groups.uni-paderborn.de/kat/BearingDataCenter/{label}.rar",
            cache / "PU" / f"{label}.rar",
        )
        with tempfile.TemporaryDirectory(prefix=f".{label}-", dir=str(target.parent)) as temp_name:
            extracted = Path(temp_name) / "extracted"
            extracted.mkdir()
            command = _rar_command(archive, extracted, "PU")
            print(f"Extracting PU {label} with {Path(command[0]).name}")
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise DatasetDownloadError(f"Failed to extract {archive}: {exc}") from exc

            candidates = [
                item for item in extracted.rglob(label)
                if item.is_dir() and len(list(item.glob("*.mat"))) >= 80
            ]
            if len(candidates) != 1:
                raise DatasetDownloadError(
                    f"Expected one {label} directory with 80 MAT files, found {len(candidates)}."
                )
            os.replace(candidates[0], label_target)
        _cleanup_archive(archive, keep_archive)


def _seven_zip_executable(dataset_name: str) -> str:
    """Return a 7-Zip executable required for nested 7z archives."""

    seven_zip = shutil.which("7zz") or shutil.which("7z") or shutil.which("7z.exe")
    if seven_zip:
        return seven_zip
    raise DatasetDownloadError(
        f"{dataset_name} contains a nested 7z archive. Install 7-Zip/7zz "
        "and run the command again."
    )


def _copy_ims_timestamp_files(source_root: Path, destination: Path) -> int:
    """Copy IMS timestamp recordings from an extracted subtree into one flat folder."""

    count = 0
    for source_file in source_root.rglob("*"):
        if not source_file.is_file() or not _IMS_TIMESTAMP.fullmatch(source_file.name):
            continue
        target = destination / source_file.name
        if target.exists():
            raise DatasetDownloadError(f"Duplicate IMS recording after extraction: {source_file.name}")
        shutil.copy2(source_file, target)
        count += 1
    return count


def _extract_ims_subset(archive: Path, staged: Path) -> None:
    """Extract all three IMS tests from NASA's nested Bearings archive.

    The current NASA/PHM ``4. Bearings.zip`` contains ``IMS.7z`` rather than
    the timestamp recordings directly.  ``IMS.7z`` in turn contains the
    individual ``*_test.rar`` archives.  All three official test subsets are
    expanded so the on-disk dataset is complete, even though the current LADDER
    experiment uses only ``2nd_test`` and ``3rd_test``.
    """

    if not zipfile.is_zipfile(archive):
        raise DatasetDownloadError(f"The downloaded IMS file is not a valid ZIP archive: {archive}")

    outputs = {
        "1st_test": staged / "1st_test",
        "2nd_test": staged / "2nd_test",
        "3rd_test": staged / "3rd_test",
    }
    for directory in outputs.values():
        directory.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix=".IMS-unpack-", dir=str(staged.parent)) as temp_name:
        temp = Path(temp_name)
        nested_7z = temp / "IMS.7z"

        with zipfile.ZipFile(archive) as source:
            candidates = []
            for info in source.infolist():
                member = _validate_zip_member(info)
                if not info.is_dir() and member.name.lower() == "ims.7z":
                    candidates.append(info)

            if len(candidates) != 1:
                names = [PurePosixPath(info.filename.replace("\\", "/")).name for info in source.infolist()]
                sample = ", ".join(names[:12])
                raise DatasetDownloadError(
                    "The NASA Bearings ZIP did not contain exactly one IMS.7z archive "
                    f"(found {len(candidates)}). First archive entries: {sample}"
                )

            print("Extracting nested IMS.7z from the NASA Bearings ZIP")
            with source.open(candidates[0]) as input_file, nested_7z.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=_CHUNK_SIZE)

        seven_zip = _seven_zip_executable("IMS")
        ims_container = temp / "ims_container"
        ims_container.mkdir()
        print(f"Extracting IMS.7z with {Path(seven_zip).name}")
        try:
            subprocess.run(
                [seven_zip, "x", "-y", f"-o{ims_container}", str(nested_7z)],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DatasetDownloadError(f"Failed to extract nested IMS.7z: {exc}") from exc

        counts: Dict[str, int] = {}
        for subset in ("1st_test", "2nd_test", "3rd_test"):
            # The canonical IMS.7z packages each test as a RAR archive.  Use a
            # recursive search so an additional top-level directory does not
            # break the downloader.
            rar_candidates = [
                item for item in ims_container.rglob(f"{subset}.rar") if item.is_file()
            ]

            extracted_subset = temp / f"extracted_{subset}"
            extracted_subset.mkdir()

            if len(rar_candidates) == 1:
                command = _rar_command(rar_candidates[0], extracted_subset, "IMS")
                print(f"Extracting IMS {subset}.rar with {Path(command[0]).name}")
                try:
                    subprocess.run(command, check=True)
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise DatasetDownloadError(
                        f"Failed to extract IMS {subset}.rar: {exc}"
                    ) from exc
                search_root = extracted_subset
            elif len(rar_candidates) == 0:
                # Be tolerant of mirrors that already expand the test folder
                # inside IMS.7z instead of storing a RAR.
                dir_candidates = [
                    item for item in ims_container.rglob(subset) if item.is_dir()
                ]
                if len(dir_candidates) != 1:
                    raise DatasetDownloadError(
                        f"Could not uniquely locate IMS {subset}.rar or an extracted "
                        f"{subset} directory inside IMS.7z. Found "
                        f"{len(rar_candidates)} RAR files and {len(dir_candidates)} directories."
                    )
                search_root = dir_candidates[0]
            else:
                raise DatasetDownloadError(
                    f"Found more than one IMS archive named {subset}.rar inside IMS.7z: "
                    f"{rar_candidates}"
                )

            counts[subset] = _copy_ims_timestamp_files(search_root, outputs[subset])

    expected = _IMS_EXPECTED_COUNTS
    if counts != expected:
        raise DatasetDownloadError(
            f"Unexpected IMS archive contents after nested extraction: found {counts}, "
            f"expected {expected}. The NASA Bearings archive was downloaded successfully, "
            "but the extracted test sets do not match the layout expected by LADDER."
        )


def _download_ims(target: Path, cache: Path, keep_archive: bool) -> None:
    print(
        "IMS notice: NASA's Bearings ZIP contains a nested IMS.7z/RAR package; "
        "1st_test, 2nd_test, and 3rd_test are all expanded. "
        "The current LADDER experiment still uses only 2nd_test and 3rd_test."
    )

    # Support upgrading a dataset prepared by an older downloader that kept
    # only 2nd_test and 3rd_test.  Complete existing subsets are preserved; an
    # incomplete non-empty subset is never overwritten silently.
    missing_subsets: List[str] = []
    if target.exists():
        if not target.is_dir():
            raise DatasetDownloadError(f"IMS target exists but is not a directory: {target}")

        unexpected = [
            item.name for item in target.iterdir()
            if item.name not in _IMS_EXPECTED_COUNTS
        ]
        if unexpected:
            raise DatasetDownloadError(
                f"Unexpected entries in {target}: {unexpected}. Refusing to mix IMS layouts."
            )

        for subset, expected_count in _IMS_EXPECTED_COUNTS.items():
            directory = target / subset
            if not directory.exists():
                missing_subsets.append(subset)
                continue
            if not directory.is_dir():
                raise DatasetDownloadError(
                    f"IMS subset path exists but is not a directory: {directory}"
                )
            count = sum(
                1 for item in directory.iterdir()
                if item.is_file() and _IMS_TIMESTAMP.fullmatch(item.name)
            )
            if count == expected_count:
                print(f"IMS {subset}: already present ({count} files)")
                continue
            if any(directory.iterdir()):
                raise DatasetDownloadError(
                    f"IMS subset {directory} is incomplete: found {count} timestamp files, "
                    f"expected {expected_count}. Move it aside or complete it manually before retrying."
                )
            directory.rmdir()
            missing_subsets.append(subset)
    else:
        missing_subsets = list(_IMS_EXPECTED_COUNTS)

    if not missing_subsets:
        return

    # NASA's current PCoE repository serves the IMS Bearings archive through
    # the PHM Society S3 mirror.  Keep a distinct cache filename so old legacy
    # IMS.zip downloads are never silently reused.
    archive = _download_url(
        os.getenv(
            "LADDER_IMS_URL",
            "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip",
        ),
        cache / "NASA-IMS-Bearings.zip",
    )
    with tempfile.TemporaryDirectory(prefix=".IMS-", dir=str(target.parent)) as temp_name:
        staged = Path(temp_name) / "IMS"
        staged.mkdir()
        _extract_ims_subset(archive, staged)
        if not _ims_ready(staged):
            raise DatasetDownloadError("IMS extraction did not produce the expected file counts.")

        if target.exists():
            # Incremental upgrade: install only missing subsets and leave valid
            # existing subsets byte-for-byte untouched.
            for subset in missing_subsets:
                source = staged / subset
                destination = target / subset
                if destination.exists():
                    raise DatasetDownloadError(
                        f"Refusing to overwrite existing IMS subset during upgrade: {destination}"
                    )
                os.replace(source, destination)
        else:
            _install_directory(staged, target)

    if not _ims_ready(target):
        raise DatasetDownloadError(
            f"IMS extraction completed but the final dataset is still incomplete at {target}."
        )
    _cleanup_archive(archive, keep_archive)


_DOWNLOADERS: Dict[str, Callable[[Path, Path, bool], None]] = {
    "CWRU_10": _download_cwru,
    "XJTU": _download_xjtu,
    "SEU": _download_seu,
    "MFPT": _download_mfpt,
    "PU": _download_pu,
    "IMS": _download_ims,
}


def ensure_dataset_available(
    dataset_name: str,
    src_dir: PathLike,
    *,
    auto_download: Optional[bool] = None,
) -> Path:
    """Ensure one selected dataset exists in the layout consumed by LADDER.

    Existing complete datasets are never downloaded again.  Existing incomplete
    directories are not deleted or overwritten.
    """

    name = _canonical_dataset_name(dataset_name)
    target = dataset_path(name, src_dir)
    if _READY_CHECKS[name](target):
        return target

    enabled = _env_flag("LADDER_AUTO_DOWNLOAD", True) if auto_download is None else auto_download
    if not enabled:
        raise FileNotFoundError(
            f"Dataset {name} is missing or incomplete at {target}. "
            f"Run 'python -m utils.download_datasets --datasets {name}' or enable "
            "LADDER_AUTO_DOWNLOAD."
        )

    datasets_root = target.parent
    datasets_root.mkdir(parents=True, exist_ok=True)
    cache = datasets_root / ".downloads"
    cache.mkdir(parents=True, exist_ok=True)
    keep_archive = _env_flag("LADDER_KEEP_DATASET_ARCHIVES", False)

    print(f"Dataset {name} is missing; preparing it at {target}")
    _DOWNLOADERS[name](target, cache, keep_archive)
    if not _READY_CHECKS[name](target):
        raise DatasetDownloadError(
            f"Dataset {name} was downloaded but its layout is still invalid at {target}."
        )
    print(f"Dataset {name} is ready at {target}")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download datasets used by LADDER")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=f"datasets to prepare: {', '.join(SUPPORTED_DATASETS)}",
    )
    parser.add_argument(
        "--src-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="path to the project's src directory",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only report whether each dataset is complete",
    )
    parser.add_argument(
        "--cwru-variant",
        choices=_CWRU_VARIANTS,
        default=None,
        help=(
            "CWRU layout: legacy reproduces the historical LADDER file "
            "(default); official keeps CWRU's unmodified 99.mat"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cwru_variant is not None:
        os.environ[_CWRU_VARIANT_ENV] = args.cwru_variant
    failed = False
    for requested_name in args.datasets:
        try:
            name = _canonical_dataset_name(requested_name)
            target = dataset_path(name, args.src_dir)
            if args.check_only:
                state = "ready" if _READY_CHECKS[name](target) else "missing/incomplete"
                print(f"{name}: {state} ({target})")
            else:
                ensure_dataset_available(name, args.src_dir, auto_download=True)
        except (DatasetDownloadError, FileNotFoundError) as exc:
            failed = True
            print(f"ERROR: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
