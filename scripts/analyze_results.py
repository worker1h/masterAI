import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import CHANNELS,ImpactMeshDataset
from src.metrics import binary_metrics
from src.model import build_model


EXPERIMENTS=[("E0","configs/stage1_e0.yaml"),("E1","configs/stage1_e1.yaml"),("E2","configs/stage1_e2.yaml"),("E3","configs/stage1_e3.yaml")]


def load(name,config_path,device):
    cfg=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); ckpt=torch.load(Path(cfg["output_dir"])/"best.pt",map_location=device,weights_only=False)
    model=build_model(cfg.get("model","unet"),CHANNELS[cfg["input_mode"]],cfg["base_channels"]).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    run=json.loads((Path(cfg["output_dir"])/"run.json").read_text(encoding="utf-8")); events=set(run["held_out_events"])
    all_ds=ImpactMeshDataset(cfg["data_root"],cfg["val_split"],cfg["input_mode"]); names=[n for n in all_ds.names if n.split("_",1)[0] in events]
    ds=ImpactMeshDataset(cfg["data_root"],cfg["val_split"],cfg["input_mode"],sample_names=names)
    return cfg,model,ds,ckpt["metrics"]


def main():
    out=Path("outputs/stage1_summary"); out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded={name:load(name,path,device) for name,path in EXPERIMENTS}; rows=[]; per_event=[]
    for name,(cfg,model,ds,best) in loaded.items():
        rows.append({"experiment":name,"input":cfg["input_mode"],**best})
        counts=defaultdict(lambda:np.zeros(4,dtype=np.float64))
        with torch.no_grad():
            for x,y,sample in ds:
                p=torch.sigmoid(model(x[None].to(device)))[0,0].cpu()>=.5; t=y[0]>=.5; event=sample.split("_",1)[0]
                counts[event]+=[torch.logical_and(p,t).sum(),torch.logical_and(p,~t).sum(),torch.logical_and(~p,t).sum(),torch.logical_and(~p,~t).sum()]
        for event,c in sorted(counts.items()): per_event.append({"experiment":name,"event":event,**binary_metrics(*c)})
    for filename,data in (("ablation.csv",rows),("per_event_metrics.csv",per_event)):
        with (out/filename).open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    fig,ax=plt.subplots(figsize=(8,4.8)); x=np.arange(4); width=.36
    ax.bar(x-width/2,[r["iou"] for r in rows],width,label="IoU"); ax.bar(x+width/2,[r["dice"] for r in rows],width,label="Dice/F1")
    ax.set_xticks(x,[r["experiment"] for r in rows]); ax.set_ylim(0,.75); ax.set_ylabel("score"); ax.set_title("ImpactMesh-Flood stage-1 event-held-out ablation"); ax.legend(); ax.grid(axis="y",alpha=.25)
    for bars in ax.containers: ax.bar_label(bars,fmt="%.3f",padding=2)
    fig.tight_layout(); fig.savefig(out/"ablation.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,(name,path) in zip(axes.flat,EXPERIMENTS):
        cfg=yaml.safe_load(Path(path).read_text(encoding="utf-8")); hist=list(csv.DictReader((Path(cfg["output_dir"])/"metrics.csv").open(encoding="utf-8")))
        ax.plot([int(r["epoch"]) for r in hist],[float(r["iou"]) for r in hist],marker="o",ms=3,label="IoU"); ax.plot([int(r["epoch"]) for r in hist],[float(r["dice"]) for r in hist],label="Dice"); ax.set_title(name); ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Validation curves"); fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=180); plt.close(fig)
    # One representative tile from each of eight held-out events, shared across E0-E3.
    base_ds=loaded["E2"][2]; best_by_event={}
    for idx,sample in enumerate(base_ds.names):
        event=sample.split("_",1)[0]; _,mask,_=base_ds[idx]; score=float(mask.sum())
        if event not in best_by_event or score>best_by_event[event][0]: best_by_event[event]=(score,sample)
    chosen=[v[1] for _,v in sorted(best_by_event.items())[:8]]
    comparison_dir=out/"comparisons"; comparison_dir.mkdir(exist_ok=True)
    for old in comparison_dir.glob("*.png"): old.unlink()
    maps={name:{n:i for i,n in enumerate(ds.names)} for name,(_,_,ds,_) in loaded.items()}
    for sample in chosen:
        e2x,y,_=loaded["E2"][2][maps["E2"][sample]]; panels=[e2x[0].numpy(),e2x[2].numpy(),y[0].numpy()]; titles=["pre-event VV","event VV","truth"]
        for name,(cfg,model,ds,_) in loaded.items():
            x,_,_=ds[maps[name][sample]]
            with torch.no_grad(): panels.append(torch.sigmoid(model(x[None].to(device)))[0,0].cpu().numpy())
            titles.append(name+" probability")
        fig,axes=plt.subplots(2,4,figsize=(13,6.5)); cmaps=["gray","gray","Blues"]+["viridis"]*4
        for ax in axes.flat: ax.axis("off")
        for ax,title,array,cmap in zip(axes.flat,titles,panels,cmaps): ax.imshow(array,cmap=cmap); ax.set_title(title); ax.axis("off")
        fig.suptitle(sample); fig.tight_layout(); fig.savefig(comparison_dir/f"{sample}.png",dpi=150); plt.close(fig)
    (out/"summary.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__": main()
