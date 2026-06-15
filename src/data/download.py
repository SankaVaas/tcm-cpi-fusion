"""
Download and cache all public datasets.
All sources are open-access — no credentials required.
"""
from pathlib import Path
import requests
from tqdm import tqdm

RAW = Path("data/raw")

SOURCES = {
    "bindingdb": (
        "https://www.bindingdb.org/bind/downloads/BindingDB_All_2D_v2.tsv.zip",
        RAW / "bindingdb" / "BindingDB_All_2D_v2.tsv.zip",
    ),
}

def download_file(url: str, dest: Path, chunk_size: int = 8192):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

def download_all():
    for name, (url, dest) in SOURCES.items():
        if dest.exists():
            print(f"[skip] {name} already cached at {dest}")
            continue
        print(f"[download] {name}")
        download_file(url, dest)
    print("Done. Remember to manually export TCMSP CSVs — see docs/DATA_SOURCES.md")

if __name__ == "__main__":
    download_all()
