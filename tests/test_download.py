"""Download helpers, tested with a fake opener: no network access."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from mtc.data.download import (
    RemoteFile,
    download_file,
    list_record_files,
    md5sum,
    select_files,
)

PAYLOAD = b"not really a tarball"
PAYLOAD_MD5 = hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()
REMOTE = RemoteFile("a.tar.gz", len(PAYLOAD), PAYLOAD_MD5, "https://example.test/a")


class FakeOpener:
    def __init__(self, body: bytes):
        self.body = body
        self.urls: list[str] = []

    def __call__(self, url: str):
        self.urls.append(url)
        return io.BytesIO(self.body)


def test_list_record_files_parses_zenodo_record():
    record = {
        "files": [
            {"key": "a.tar.gz", "size": 20, "checksum": "md5:abc", "links": {"self": "https://x/a"}}
        ]
    }
    opener = FakeOpener(json.dumps(record).encode())

    files = list_record_files("123", "https://zenodo.test/api/", opener)

    assert files == [RemoteFile("a.tar.gz", 20, "abc", "https://x/a")]
    assert opener.urls == ["https://zenodo.test/api/records/123"]


def test_md5sum(tmp_path: Path):
    path = tmp_path / "f"
    path.write_bytes(PAYLOAD)
    assert md5sum(path) == PAYLOAD_MD5


def test_download_file_writes_and_verifies(tmp_path: Path):
    target = download_file(REMOTE, tmp_path / "raw", FakeOpener(PAYLOAD))

    assert target.read_bytes() == PAYLOAD
    assert not target.with_name("a.tar.gz.part").exists()


def test_download_file_skips_existing(tmp_path: Path):
    (tmp_path / "a.tar.gz").write_bytes(PAYLOAD)
    opener = FakeOpener(PAYLOAD)

    download_file(REMOTE, tmp_path, opener)

    assert opener.urls == []


def test_download_file_rejects_bad_md5(tmp_path: Path):
    with pytest.raises(ValueError, match="md5 mismatch"):
        download_file(REMOTE, tmp_path, FakeOpener(b"corrupted"))

    assert list(tmp_path.iterdir()) == []


def test_select_files():
    other = RemoteFile("b.tar.gz", 1, "x", "https://example.test/b")

    assert select_files([REMOTE, other], None) == [REMOTE, other]
    assert select_files([REMOTE, other], ["b.tar.gz"]) == [other]
    with pytest.raises(ValueError, match="not in the Zenodo record"):
        select_files([REMOTE], ["missing.tar.gz"])
