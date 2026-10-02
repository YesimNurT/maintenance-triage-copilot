"""Download the NGAFID maintenance dataset archives from Zenodo.

The file list, sizes and md5 checksums come from the Zenodo record API, so nothing
about the archives is hard-coded here. Files that are already present with the right
size are skipped; every download is verified against its md5 before it is kept.
"""

import hashlib
import json
import logging
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20

Opener = Callable[[str], BinaryIO]


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int
    md5: str
    url: str


def _urlopen(url: str) -> BinaryIO:
    return urllib.request.urlopen(url)  # noqa: S310 - https URL built from settings


def list_record_files(record_id: str, api_url: str, opener: Opener = _urlopen) -> list[RemoteFile]:
    """Return the files of a Zenodo record with their size, md5 and download URL."""
    with opener(f"{api_url.rstrip('/')}/records/{record_id}") as response:
        record = json.load(response)
    return [
        RemoteFile(
            name=f["key"],
            size=int(f["size"]),
            md5=f["checksum"].removeprefix("md5:"),
            url=f["links"]["self"],
        )
        for f in record["files"]
    ]


def md5sum(path: Path) -> str:
    """md5 of a file, read in chunks so large archives do not fill memory."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(remote: RemoteFile, dest_dir: Path, opener: Opener = _urlopen) -> Path:
    """Download one file into ``dest_dir`` and verify its md5.

    Skips the download when the target already exists with the expected size. Writes to
    a ``.part`` file first so an interrupted run never leaves a truncated archive behind.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / remote.name
    if target.exists() and target.stat().st_size == remote.size:
        logger.info("skip %s (already downloaded)", remote.name)
        return target

    partial = target.with_name(target.name + ".part")
    logger.info("downloading %s (%.2f GB)", remote.name, remote.size / 1e9)
    with opener(remote.url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK_BYTES)

    actual = md5sum(partial)
    if actual != remote.md5:
        partial.unlink()
        raise ValueError(f"md5 mismatch for {remote.name}: expected {remote.md5}, got {actual}")
    partial.replace(target)
    return target


def select_files(files: list[RemoteFile], names: list[str] | None) -> list[RemoteFile]:
    """Keep only the requested file names (all files when ``names`` is empty)."""
    if not names:
        return files
    available = {f.name for f in files}
    missing = sorted(set(names) - available)
    if missing:
        raise ValueError(f"not in the Zenodo record: {missing}; available: {sorted(available)}")
    return [f for f in files if f.name in names]
