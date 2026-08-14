import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


BASE="https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood/resolve/main"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--split",default="val"); ap.add_argument("--modalities",nargs="+",default=["MASK","DEM","S1RTC"]); ap.add_argument("--root",default="data/raw"); args=ap.parse_args()
    for mod in args.modalities:
        dest=Path(args.root)/args.split/mod; dest.mkdir(parents=True,exist_ok=True)
        archive=dest.parent/f"{mod}.tar"
        if not archive.exists():
            url=f"{BASE}/{args.split}/{mod}.tar?download=true"; print(f"downloading {url}")
            with urllib.request.urlopen(url) as src, archive.open("wb") as dst: shutil.copyfileobj(src,dst,16*1024*1024)
        with tarfile.open(archive) as tf: tf.extractall(dest,filter="data")
        print(mod, len(list(dest.rglob("*.zarr.zip"))), "samples")


if __name__ == "__main__": main()

