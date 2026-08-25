"""Download and extract selected Global Flood Forecasting (GFF) components.

The official Zenodo endpoint can be slow for a single connection.  This helper
uses resumable HTTP range requests, verifies the published MD5 checksum, and
then safely extracts each archive into one shared GFF data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable
import zipfile

import requests


RECORD_ID = 14184289
ZENODO_RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
DEFAULT_COMPONENTS = (
    "base",
    "era5",
    "glofas",
    "hydroatlas",
    "dem",
    "hand",
    "s1",
    "worldcover",
)


def public_download_url(key: str) -> str:
    """Use the public file route; the API content route fails on high byte ranges."""

    return f"https://zenodo.org/records/{RECORD_ID}/files/{quote(key)}?download=1"


@dataclass(frozen=True)
class RemoteFile:
    key: str
    size: int
    md5: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/gff"),
        help="Shared extraction directory (archives go in ROOT/downloads).",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(DEFAULT_COMPONENTS),
        help="Component names without .zip.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--part-mb",
        type=int,
        default=16,
        help="Range size. Zenodo is substantially more reliable around 16 MiB.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=100,
        help="Consecutive zero-progress failures allowed per range.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download and verify archives.",
    )
    return parser.parse_args()


def fetch_manifest(cache_path: Path | None = None) -> dict[str, RemoteFile]:
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Using cached Zenodo manifest: {cache_path}", flush=True)
        return {
            key: RemoteFile(
                key=key,
                size=int(value["size"]),
                md5=str(value["md5"]).lower(),
                url=public_download_url(key),
            )
            for key, value in cached.items()
        }
    response = requests.get(ZENODO_RECORD_API, timeout=60)
    response.raise_for_status()
    record = response.json()
    manifest: dict[str, RemoteFile] = {}
    for item in record["files"]:
        checksum = item["checksum"]
        algorithm, value = checksum.split(":", maxsplit=1)
        if algorithm.lower() != "md5":
            raise ValueError(f"Unsupported checksum for {item['key']}: {checksum}")
        manifest[item["key"]] = RemoteFile(
            key=item["key"],
            size=int(item["size"]),
            md5=value.lower(),
            url=public_download_url(item["key"]),
        )
    return manifest


def file_md5(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def part_ranges(size: int, part_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(size - 1, start + part_size - 1))
        for start in range(0, size, part_size)
    ]


def download_part(
    remote: RemoteFile,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
) -> int:
    expected = end - start + 1
    part_path.parent.mkdir(parents=True, exist_ok=True)

    current = part_path.stat().st_size if part_path.exists() else 0
    if current == expected:
        return expected
    if current > expected:
        part_path.unlink()
        current = 0

    # Zenodo commonly closes a long range response after transferring a useful
    # prefix. Treat that as progress rather than consuming the retry budget;
    # ``retries`` counts only consecutive attempts that add no bytes.
    failures_without_progress = 0
    while current < expected:
        before_attempt = current
        try:
            headers = {"Range": f"bytes={start + current}-{end}"}
            with requests.get(
                remote.url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Range request returned HTTP {response.status_code}"
                    )
                with part_path.open("ab") as output:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
            current = part_path.stat().st_size
            if current == expected:
                return expected
            if current > expected:
                raise RuntimeError(f"Part grew beyond expected size: {part_path}")
        except Exception as exc:  # noqa: BLE001 - retries are intentional here
            current = part_path.stat().st_size if part_path.exists() else 0
            if current > before_attempt:
                failures_without_progress = 0
            else:
                failures_without_progress += 1
            if failures_without_progress >= retries:
                raise RuntimeError(
                    f"Failed range {start}-{end} for {remote.key}: {exc}"
                ) from exc
            delay = 1 if current > before_attempt else min(60, 2**failures_without_progress)
            if current == before_attempt:
                print(
                    f"Retry {remote.key} range {start}-{end} at "
                    f"{current}/{expected} bytes; consecutive failures "
                    f"{failures_without_progress}/{retries} in {delay}s: {exc}",
                    flush=True,
                )
            time.sleep(delay)
    return current


def assemble(remote: RemoteFile, archive: Path, parts: Iterable[Path]) -> None:
    temp_path = archive.with_suffix(archive.suffix + ".assembling")
    with temp_path.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if temp_path.stat().st_size != remote.size:
        raise RuntimeError(
            f"Assembled size mismatch for {remote.key}: "
            f"{temp_path.stat().st_size} != {remote.size}"
        )
    checksum = file_md5(temp_path)
    if checksum != remote.md5:
        raise RuntimeError(
            f"MD5 mismatch for {remote.key}: {checksum} != {remote.md5}"
        )
    os.replace(temp_path, archive)


def download(remote: RemoteFile, downloads: Path, workers: int, part_mb: int, retries: int) -> Path:
    archive = downloads / remote.key
    if archive.exists() and archive.stat().st_size == remote.size:
        checksum = file_md5(archive)
        if checksum == remote.md5:
            print(f"Verified existing {remote.key}", flush=True)
            return archive

    if archive.exists():
        incomplete = archive.with_suffix(archive.suffix + ".single.incomplete")
        if not incomplete.exists():
            archive.replace(incomplete)
        else:
            archive.unlink()

    # Keep enough parts to saturate the requested workers for small archives,
    # while capping request count for the multi-gigabyte modalities.
    max_part_size = part_mb * 1024 * 1024
    parallel_part_size = max(8 * 1024 * 1024, math.ceil(remote.size / workers))
    part_size = min(max_part_size, parallel_part_size)
    ranges = part_ranges(remote.size, part_size)
    part_dir = downloads / ".parts" / remote.key
    part_paths = [part_dir / f"{start:020d}-{end:020d}.part" for start, end in ranges]
    print(
        f"Downloading {remote.key}: {remote.size / 2**30:.2f} GiB "
        f"in {len(ranges)} resumable parts with {workers} workers",
        flush=True,
    )
    completed_bytes = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_part, remote, path, start, end, retries): index
            for index, (path, (start, end)) in enumerate(zip(part_paths, ranges), start=1)
        }
        completed_parts = 0
        for future in as_completed(futures):
            completed_bytes += future.result()
            completed_parts += 1
            elapsed = max(1e-6, time.monotonic() - started)
            speed = completed_bytes / elapsed / 2**20
            percent = completed_bytes / remote.size * 100
            print(
                f"  {remote.key}: {completed_parts}/{len(ranges)} parts, "
                f"{percent:5.1f}%, {speed:.2f} MiB/s",
                flush=True,
            )

    assemble(remote, archive, part_paths)
    shutil.rmtree(part_dir)
    print(f"Verified MD5 for {remote.key}: {remote.md5}", flush=True)
    return archive


def safe_extract(archive: Path, root: Path) -> None:
    root_resolved = root.resolve()
    print(f"Extracting {archive.name} into {root_resolved}", flush=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if root_resolved != target and root_resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(root)


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.part_mb < 1:
        raise ValueError("workers and part-mb must be positive")

    root = args.root.resolve()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "zenodo_manifest.json"
    manifest = fetch_manifest(manifest_path)
    manifest_path.write_text(
        json.dumps(
            {
                key: {"size": value.size, "md5": value.md5, "url": value.url}
                for key, value in manifest.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    requested = [f"{name}.zip" if not name.endswith(".zip") else name for name in args.components]
    missing = sorted(set(requested) - set(manifest))
    if missing:
        raise KeyError(f"Unknown GFF components: {missing}")

    total = sum(manifest[key].size for key in requested)
    print(
        f"GFF Zenodo record {RECORD_ID}; selected {len(requested)} archives, "
        f"{total / 2**30:.2f} GiB compressed",
        flush=True,
    )
    for key in requested:
        archive = download(
            manifest[key], downloads, args.workers, args.part_mb, args.retries
        )
        if not args.no_extract:
            safe_extract(archive, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
