import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{code}.rar"
BEARINGS = ["K001", "KA01", "KI01"]
CHUNK = 1024 * 1024
OUT = Path("data/raw")


def download(url, dest):
    tmp = dest.with_suffix(".part")
    try:
        resp = urllib.request.urlopen(url)
    except urllib.error.HTTPError as err:
        sys.exit(f"Server answered {err.code} for {url}. Try again later.")
    if "html" in resp.headers.get("Content-Type", ""):
        sys.exit(f"Got a web page instead of an archive from {url}.")
    with resp, open(tmp, "wb") as f:
        while chunk := resp.read(CHUNK):
            f.write(chunk)
        tmp.rename(dest)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for code in BEARINGS:
        archive = OUT / f"{code}.rar"
        if not archive.exists():
            print(f"{code}: downloading")
            download(BASE_URL.format(code=code), archive)
        if not (OUT / code).is_dir():
            print(f"{code}: extracting")
            subprocess.run(["unrar", "-q", "-o",  str(OUT), str(archive)], check=True)
            n_files = len(list((OUT / code).glob("*.mat")))
            print(f"{code}: {n_files} .mat files")


if __name__ == "__main__":
    main()