import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import CHANNELS,FutureEventForecastDataset,ImpactMeshDataset
from src.metrics import binary_metrics
from src.model import build_model


def boundary_counts(prediction, target, tolerance=2):
    """Return matched/predicted/true boundary pixels with a tolerance in pixels."""
    prediction=prediction.float();target=target.float()
    pred_boundary=(F.max_pool2d(prediction,3,1,1)+F.max_pool2d(-prediction,3,1,1)>0)
    true_boundary=(F.max_pool2d(target,3,1,1)+F.max_pool2d(-target,3,1,1)>0)
    kernel=2*tolerance+1
    pred_near=F.max_pool2d(pred_boundary.float(),kernel,1,tolerance)>0
    true_near=F.max_pool2d(true_boundary.float(),kernel,1,tolerance)>0
    matched_pred=torch.logical_and(pred_boundary,true_near).sum().item()
    matched_true=torch.logical_and(true_boundary,pred_near).sum().item()
    return matched_pred,pred_boundary.sum().item(),matched_true,true_boundary.sum().item()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--split",default="test");ap.add_argument("--sample-list");ap.add_argument("--name",default="test");args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); names=None
    if args.sample_list: names=[x.strip() for x in Path(args.sample_list).read_text(encoding="utf-8").splitlines() if x.strip()]
    if cfg.get("task") == "future_event_forecast": ds=FutureEventForecastDataset(cfg["manifest"],args.split)
    else: ds=ImpactMeshDataset(cfg["data_root"],args.split,cfg["input_mode"],sample_names=names)
    loader=DataLoader(ds,batch_size=cfg["batch_size"],num_workers=cfg["num_workers"],pin_memory=True)
    input_channels=int(cfg.get("input_channels",CHANNELS[cfg["input_mode"]]))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); state=torch.load(Path(cfg["output_dir"])/"best.pt",map_location=device,weights_only=False);model=build_model(cfg.get("model","unet"),input_channels,cfg["base_channels"]).to(device);model.load_state_dict(state["model"]);model.eval(); counts=defaultdict(lambda:np.zeros(4));boundary=np.zeros(4,dtype=np.float64)
    with torch.no_grad():
        for x,y,samples in loader:
            x=x.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"): p=torch.sigmoid(model(x))>=.5
            t=y.to(device)>=.5
            b=boundary_counts(p,t);boundary+=b
            for i,sample in enumerate(samples):
                event=sample.split("_",1)[0];counts[event]+=[torch.logical_and(p[i],t[i]).sum().item(),torch.logical_and(p[i],~t[i]).sum().item(),torch.logical_and(~p[i],t[i]).sum().item(),torch.logical_and(~p[i],~t[i]).sum().item()]
    total=sum(counts.values(),np.zeros(4));bp=boundary[0]/max(boundary[1],1);br=boundary[2]/max(boundary[3],1);bf=2*bp*br/max(bp+br,1e-12);summary={"name":args.name,"samples":len(ds),"events":len(counts),**binary_metrics(*total),"boundary_precision":bp,"boundary_recall":br,"boundary_f1":bf,"boundary_tolerance_pixels":2};out=Path(cfg["output_dir"]);(out/f"{args.name}_metrics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[{"event":e,"truth_positive_pixels":int(c[0]+c[2]),"predicted_positive_pixels":int(c[0]+c[1]),**binary_metrics(*c)} for e,c in sorted(counts.items())]
    with (out/f"{args.name}_per_event.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":main()
