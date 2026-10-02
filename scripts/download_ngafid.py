"""Download the NGAFID archives from Zenodo into the raw data directory.

    uv run python scripts/download_ngafid.py --list
    uv run python scripts/download_ngafid.py                    # every file in the record
    uv run python scripts/download_ngafid.py --files 2days.tar.gz
"""

import argparse
import logging

from mtc.config import get_settings
from mtc.data.download import download_file, list_record_files, select_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", nargs="*", help="file names to download (default: all)")
    parser.add_argument("--list", action="store_true", help="only print the record's files")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(message)s")

    files = list_record_files(settings.zenodo_record_id, settings.zenodo_api_url)
    if args.list:
        for f in files:
            print(f"{f.name}\t{f.size / 1e9:.2f} GB\tmd5:{f.md5}")
        return

    for remote in select_files(files, args.files):
        path = download_file(remote, settings.ngafid_raw_dir)
        print(f"ok {path}")


if __name__ == "__main__":
    main()
