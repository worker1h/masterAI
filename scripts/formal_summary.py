import csv
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


CONFIGS=[("E0",Path("configs/formal_e0.yaml")),("E1",Path("configs/formal_e1.yaml")),("E2",Path("configs/formal_e2.yaml")),("E3",Path("configs/formal_e3.yaml"))]


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    out=Path("outputs/formal_summary");out.mkdir(parents=True,exist_ok=True);rows=[]
    for name,path in CONFIGS:
        cfg=yaml.safe_load(path.read_text(encoding="utf-8"));state=torch.load(Path(cfg["output_dir"])/"best.pt",map_location="cpu",weights_only=False);test=json.loads((Path(cfg["output_dir"])/"test_holdout_metrics.json").read_text(encoding="utf-8"))
        events=list(csv.DictReader((Path(cfg["output_dir"])/"test_holdout_per_event.csv").open(encoding="utf-8-sig"))); positive=[r for r in events if int(r["truth_positive_pixels"])>0]
        rows.append({"experiment":name,"input":cfg["input_mode"],"val_iou":state["metrics"]["iou"],"val_dice":state["metrics"]["dice"],"val_precision":state["metrics"]["precision"],"val_recall":state["metrics"]["recall"],"holdout_iou":test["iou"],"holdout_dice":test["dice"],"holdout_precision":test["precision"],"holdout_recall":test["recall"],"positive_event_macro_iou":float(np.mean([float(r["iou"]) for r in positive])),"positive_event_macro_dice":float(np.mean([float(r["dice"]) for r in positive]))})
    with (out/"formal_ablation.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True);x=range(4)
    for ax,prefix,title in ((axes[0],"val_","Official validation"),(axes[1],"holdout_","Unseen-event test_holdout")):
        b1=ax.bar([i-.18 for i in x],[r[prefix+"iou"] for r in rows],.36,label="IoU");b2=ax.bar([i+.18 for i in x],[r[prefix+"dice"] for r in rows],.36,label="Dice/F1");ax.set_xticks(list(x),[r["experiment"] for r in rows]);ax.set_title(title);ax.grid(axis="y",alpha=.25);ax.legend();ax.bar_label(b1,fmt="%.3f",fontsize=8);ax.bar_label(b2,fmt="%.3f",fontsize=8)
    axes[0].set_ylabel("score");axes[0].set_ylim(0,.75);fig.suptitle("ImpactMesh-Flood formal ablation (seed 42)");fig.tight_layout();fig.savefig(out/"formal_ablation.png",dpi=180);plt.close(fig)
    manifests=[]
    for split,expected in (("train",19448),("val",2171),("test",2959)):
        root=Path("data/raw")/split;manifests.append({"split":split,"expected":expected,"S1RTC":len(list((root/"S1RTC").rglob("*_S1RTC.zarr.zip"))),"DEM":len(list((root/"DEM").rglob("*_DEM.tif"))),"MASK":len(list((root/"MASK").rglob("*_annotation_flood.tif")))})
    split_files=[]
    for path in sorted(Path("data/split").glob("*.txt")):split_files.append({"file":str(path),"lines":sum(1 for x in path.read_text(encoding="utf-8").splitlines() if x.strip()),"sha256":sha256(path)})
    (out/"data_manifest.json").write_text(json.dumps({"dataset":"ibm-esa-geospatial/ImpactMesh-Flood","version":"v1 main snapshot 2026-08-12","modalities":manifests,"split_files":split_files},ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"formal_summary.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
