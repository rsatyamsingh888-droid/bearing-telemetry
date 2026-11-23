import re
import sys
import zipfile
from pathlib import Path
import shutil

OUT = Path("data/raw")
PATTERN = re.compile(r"([HIOBC])_(\d+)_([012])\.csv$")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/prepare_data.py <path to downloaded zip>")
    archive = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)

    if archive.is_dir():
        for path in archive.rglob("*.csv"):
            if "1_CSV_Raw_Data_Files" not in str(path):
                continue
            match = PATTERN.search(path.name)
            if match and not (OUT / path.name).exists():
                shutil.copy2(path, OUT / path.name)
    else:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if "1_CSV_Raw_Data_Files" not in name:
                    continue
                match = PATTERN.search(name)
                if match is None:
                    continue
                dest = OUT / match.group(0)
                if not dest.exists():
                    dest.write_bytes(z.read(name))

    files = sorted(OUT.glob("*.csv"))
    print(f"{len(files)} CSV files in {OUT}")

    by_bearing = {}
    for f in files:
        kind, bearing, state = PATTERN.search(f.name).groups()
        by_bearing.setdefault(int(bearing), []).append(f"{kind}{state}")

    for bearing in sorted(by_bearing):
        print(f"bearing {bearing:2d}: {' '.join(sorted(by_bearing[bearing]))}")


if __name__ == "__main__":
    main()