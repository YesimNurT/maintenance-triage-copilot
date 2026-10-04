"""Expose a downloaded NGAFID copy (e.g. /kaggle/input) in the layout the code expects.

    python scripts/kaggle_prepare_raw.py /kaggle/input

Links the files into NGAFID_RAW_DIR. If the copy only holds .tar.gz archives they are
extracted next to the links first.
"""

import argparse
from pathlib import Path

from mtc.config import Settings, get_settings
from mtc.data.layout import extract_archives, find_raw_parts, link_raw_layout


def run(settings: Settings, input_root: Path) -> Path:
    target = settings.ngafid_raw_dir
    try:
        find_raw_parts(input_root)
        source = input_root
    except FileNotFoundError:
        source = target / "_extracted"
        archives = extract_archives(input_root, source)
        print(f"extracted {[a.name for a in archives]} to {source}")
    link_raw_layout(source, target)
    for path in sorted(target.rglob("*")):
        if path.is_symlink():
            print(f"{path.relative_to(target)} -> {path.resolve()}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_root", type=Path)
    run(get_settings(), parser.parse_args().input_root)
