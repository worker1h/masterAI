from pathlib import Path
import rasterio


names=[x.strip() for x in Path("data/split/impactmesh_flood_test_holdout.txt").read_text(encoding="utf-8").splitlines() if x.strip()];root=Path("data/raw/test/MASK"); scored=[]
for name in names:
    path=next(root.rglob(name+"_annotation_flood.tif"))
    with rasterio.open(path) as src: score=int((src.read(1)==2).sum())
    scored.append((score,name))
selected=[name for score,name in sorted(scored,reverse=True)[:20]];out=Path("outputs/formal_summary/visualization_samples.txt");out.write_text("\n".join(selected)+"\n",encoding="utf-8");print(out)
