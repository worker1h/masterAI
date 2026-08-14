import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from .data import CHANNELS, ImpactMeshDataset
from .losses import segmentation_loss
from .metrics import binary_metrics
from .model import UNet
from .sampling import make_flood_fraction_sampler


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, pos_weight=None, loss_config=None):
    model.eval(); totals = np.zeros(4, dtype=np.float64); loss_sum = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                logits = model(x); loss_sum += segmentation_loss(logits, y, pos_weight, loss_config).item()
            p = torch.sigmoid(logits) >= 0.5; t = y >= 0.5
            totals += [torch.logical_and(p,t).sum().item(), torch.logical_and(p,~t).sum().item(), torch.logical_and(~p,t).sum().item(), torch.logical_and(~p,~t).sum().item()]
    return loss_sum / max(len(loader),1), binary_metrics(*totals)


def event_id(name):
    return name.split("_", 1)[0]


def event_stratified_names(dataset, seed, val_fraction=0.25):
    events=sorted({event_id(n) for n in dataset.names}); rng=random.Random(seed); rng.shuffle(events)
    n_val=max(1,round(len(events)*val_fraction)); val_events=set(events[:n_val])
    train=[n for n in dataset.names if event_id(n) not in val_events]; val=[n for n in dataset.names if event_id(n) in val_events]
    return train,val,sorted(val_events)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config", required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); seed_everything(cfg["seed"])
    out=Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    train_ds=ImpactMeshDataset(cfg["data_root"], cfg["train_split"], cfg["input_mode"], cfg.get("max_train_samples"))
    val_ds=ImpactMeshDataset(cfg["data_root"], cfg["val_split"], cfg["input_mode"], cfg.get("max_val_samples"))
    held_out_events=[]
    if cfg["train_split"] == cfg["val_split"]:
        all_ds=ImpactMeshDataset(cfg["data_root"],cfg["train_split"],cfg["input_mode"])
        train_names,val_names,held_out_events=event_stratified_names(all_ds,cfg["seed"],cfg.get("val_event_fraction",0.25))
        if cfg.get("max_train_samples") is not None: train_names=train_names[:int(cfg["max_train_samples"])]
        if cfg.get("max_val_samples") is not None: val_names=val_names[:int(cfg["max_val_samples"])]
        train_ds=ImpactMeshDataset(cfg["data_root"],cfg["train_split"],cfg["input_mode"],sample_names=train_names)
        val_ds=ImpactMeshDataset(cfg["data_root"],cfg["val_split"],cfg["input_mode"],sample_names=val_names)
    sampling_metadata={"strategy":"shuffle"}; sampler=None
    if cfg.get("sampling",{}).get("strategy") == "flood_fraction_bins":
        sampler,sampling_metadata=make_flood_fraction_sampler(
            train_ds,cfg["sampling"],cfg["seed"],out/"flood_fraction_cache.json"
        )
    train_loader=DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=sampler is None,sampler=sampler,num_workers=cfg["num_workers"],pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["num_workers"],pin_memory=torch.cuda.is_available())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNet(CHANNELS[cfg["input_mode"]],cfg["base_channels"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    rows=[]; best=-1.0
    for epoch in range(1,cfg["epochs"]+1):
        model.train(); total=0.0
        for x,y,_ in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                loss=segmentation_loss(model(x),y,cfg.get("pos_weight"),cfg.get("loss"))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()
        val_loss,m=evaluate(model,val_loader,device,cfg.get("pos_weight"),cfg.get("loss")); row={"epoch":epoch,"train_loss":total/max(len(train_loader),1),"val_loss":val_loss,**m}; rows.append(row)
        print(json.dumps(row,ensure_ascii=False))
        if m["iou"]>best:
            best=m["iou"]; torch.save({"model":model.state_dict(),"config":cfg,"metrics":m},out/"best.pt")
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out/"run.json").write_text(json.dumps({"device":str(device),"samples":{"train":len(train_ds),"val":len(val_ds)},"held_out_events":held_out_events,"best_iou":best,"sampling":sampling_metadata,"config":cfg},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
