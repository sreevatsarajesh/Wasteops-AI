import argparse
from pathlib import Path
from collections import Counter
from PIL import Image
from wasteops.taxonomy import CLASSES

ALIASES={"food organics":"food_organics","miscellaneous trash":"miscellaneous_trash","textile trash":"textile_trash"}
def find_root(base:Path):
    candidates=[p for p in base.rglob('*') if p.is_dir() and all((p / name.replace('_',' ').title()).exists() for name in [])]
    expected={"Cardboard","Food Organics","Glass","Metal","Miscellaneous Trash","Paper","Plastic","Textile Trash","Vegetation"}
    for p in [base,*[x for x in base.rglob('*') if x.is_dir()]]:
        if expected.issubset({x.name for x in p.iterdir() if x.is_dir()}): return p
    raise FileNotFoundError(f"Could not locate nine RealWaste class folders under {base}")
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default='data');args=ap.parse_args();root=find_root(Path(args.data_dir)); counts=Counter(); bad=[]
    for folder in root.iterdir():
        if not folder.is_dir():continue
        for f in folder.iterdir():
            if f.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:continue
            try:
                with Image.open(f) as im: im.verify()
                counts[folder.name]+=1
            except Exception:bad.append(str(f))
    print(f"Dataset root: {root}")
    for name,n in sorted(counts.items()):print(f"{name:24s} {n:5d}")
    print(f"Valid images: {sum(counts.values())}; unreadable: {len(bad)}")
    if len(counts)!=9: raise SystemExit("Expected nine class directories; inspect dataset layout.")
    if bad: print("Unreadable files (first 10):",bad[:10])
if __name__=='__main__':main()
