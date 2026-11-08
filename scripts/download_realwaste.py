import argparse, io, zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

def main():
    parser=argparse.ArgumentParser(description="Download official UCI RealWaste dataset")
    parser.add_argument('--output',default='data/raw/realwaste.zip')
    parser.add_argument('--data-dir',default='data')
    args=parser.parse_args(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        print(f"Downloading {URL}")
        req=Request(URL,headers={"User-Agent":"WasteOps/1.0 dataset-acquisition"})
        with urlopen(req,timeout=120) as response, out.open('wb') as f:
            while chunk:=response.read(1024*1024): f.write(chunk)
    print(f"Downloaded {out} ({out.stat().st_size/1e6:.1f} MB)")
    dest=Path(args.data_dir); dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f"Corrupt archive member: {bad}")
        z.extractall(dest)
    print(f"Extracted dataset under {dest}. Run scripts/validate_dataset.py to locate/check it.")
if __name__=='__main__': main()
