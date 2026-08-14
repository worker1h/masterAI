import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import rasterio


def event_id(name): return name.split("_",1)[0]


def main():
    root=Path("data/raw/val/MASK"); run=json.loads(Path("outputs/stage1_e2/run.json").read_text(encoding="utf-8")); val_events=set(run["held_out_events"]); counts=defaultdict(lambda:[0,0])
    for path in root.rglob("*_annotation_flood.tif"):
        event=event_id(path.name)
        with rasterio.open(path) as src: mask=src.read(1)
        counts[event][0]+=int((mask==2).sum()); counts[event][1]+=int((mask!=1).sum())
    ratios={e:f/max(land,1) for e,(f,land) in counts.items()}; train_values=np.asarray([v for e,v in ratios.items() if e not in val_events]); q1,q2=np.quantile(train_values,[1/3,2/3])
    def level(v): return "一般" if v<=q1 else ("严重" if v<=q2 else "非常严重")
    rows=[{"event":e,"split":"val_event_holdout" if e in val_events else "train_events","flood_pixels":counts[e][0],"effective_land_pixels":counts[e][1],"impact_ratio":v,"severity":level(v)} for e,v in sorted(ratios.items())]
    out=Path("outputs/severity"); out.mkdir(parents=True,exist_ok=True)
    with (out/"event_severity.csv").open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/"thresholds.json").write_text(json.dumps({"source":"training events only","quantiles":{"q33":float(q1),"q67":float(q2)},"rule":"ratio=flood pixels / pixels excluding permanent water (MASK class 1)"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(out/"event_severity.csv",q1,q2)


if __name__=="__main__": main()
