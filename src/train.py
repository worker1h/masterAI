import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from .data import CHANNELS, FutureEventForecastDataset, ImpactMeshDataset
from .losses import boundary_band, dice_loss, segmentation_loss
from .metrics import binary_metrics
from .model import build_model
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


def training_objective(model, inputs, targets, pos_weight=None, loss_config=None):
    """Return segmentation loss plus optional presence and boundary objectives."""
    loss_config = loss_config or {}
    if hasattr(model, "forward_with_aux"):
        auxiliary_outputs = model.forward_with_aux(inputs)
        logits, presence_logits = auxiliary_outputs[:2]
        segmentation = segmentation_loss(logits, targets, pos_weight, loss_config)
        presence_targets = targets.amax(dim=(1, 2, 3))
        presence = F.binary_cross_entropy_with_logits(presence_logits, presence_targets)
        loss = segmentation + float(loss_config.get("presence_weight", 0.25)) * presence
        if len(auxiliary_outputs) > 2:
            boundary_logits = auxiliary_outputs[2]
            boundary_targets = boundary_band(
                targets, int(loss_config.get("boundary_aux_kernel_size", 5))
            )
            boundary_pos_weight = torch.as_tensor(
                float(loss_config.get("boundary_aux_pos_weight", 2.0)),
                device=boundary_logits.device,
                dtype=boundary_logits.dtype,
            )
            boundary = F.binary_cross_entropy_with_logits(
                boundary_logits, boundary_targets, pos_weight=boundary_pos_weight
            ) + float(loss_config.get("boundary_aux_dice_weight", 0.5)) * dice_loss(
                boundary_logits, boundary_targets
            )
            loss = loss + float(loss_config.get("boundary_aux_weight", 0.2)) * boundary
        return loss
    return segmentation_loss(model(inputs), targets, pos_weight, loss_config)


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
    held_out_events=[]
    if cfg.get("task") == "future_event_forecast":
        train_ds = FutureEventForecastDataset(cfg["manifest"], "train")
        val_ds = FutureEventForecastDataset(cfg["manifest"], "val")
    else:
        train_ds=ImpactMeshDataset(cfg["data_root"], cfg["train_split"], cfg["input_mode"], cfg.get("max_train_samples"))
        val_ds=ImpactMeshDataset(cfg["data_root"], cfg["val_split"], cfg["input_mode"], cfg.get("max_val_samples"))
    if cfg.get("task") != "future_event_forecast" and cfg["train_split"] == cfg["val_split"]:
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
    input_channels = int(cfg.get("input_channels", CHANNELS[cfg["input_mode"]]))
    model=build_model(cfg.get("model","unet"),input_channels,cfg["base_channels"]).to(device)
    initialization={}
    if cfg.get("init_checkpoint"):
        initial_state=torch.load(cfg["init_checkpoint"],map_location=device,weights_only=False)
        incompatible=model.load_state_dict(initial_state["model"],strict=False)
        initialization={"checkpoint":cfg["init_checkpoint"],"missing_keys":incompatible.missing_keys,"unexpected_keys":incompatible.unexpected_keys}
    trainable_prefixes=cfg.get("trainable_prefixes")
    if trainable_prefixes:
        for name,parameter in model.named_parameters():
            parameter.requires_grad=any(name.startswith(prefix) for prefix in trainable_prefixes)
    trainable_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters: raise ValueError("No trainable parameters remain after applying trainable_prefixes")
    opt=torch.optim.AdamW(trainable_parameters,lr=cfg["learning_rate"])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    rows=[]; best=-1.0
    if cfg.get("evaluate_initial",False):
        val_loss,m=evaluate(model,val_loader,device,cfg.get("pos_weight"),cfg.get("loss"))
        initial_row={"epoch":0,"train_loss":None,"val_loss":val_loss,**m}; rows.append(initial_row)
        print(json.dumps(initial_row,ensure_ascii=False))
        best=m["iou"]
        torch.save({"model":model.state_dict(),"config":cfg,"metrics":m},out/"best.pt")
    for epoch in range(1,cfg["epochs"]+1):
        model.train(); total=0.0
        for x,y,_ in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                loss=training_objective(model,x,y,cfg.get("pos_weight"),cfg.get("loss"))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()
        val_loss,m=evaluate(model,val_loader,device,cfg.get("pos_weight"),cfg.get("loss")); row={"epoch":epoch,"train_loss":total/max(len(train_loader),1),"val_loss":val_loss,**m}; rows.append(row)
        print(json.dumps(row,ensure_ascii=False))
        if m["iou"]>best:
            best=m["iou"]; torch.save({"model":model.state_dict(),"config":cfg,"metrics":m},out/"best.pt")
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out/"run.json").write_text(json.dumps({"device":str(device),"samples":{"train":len(train_ds),"val":len(val_ds)},"held_out_events":held_out_events,"best_iou":best,"sampling":sampling_metadata,"initialization":initialization,"trainable_parameters":sum(parameter.numel() for parameter in trainable_parameters),"config":cfg},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
