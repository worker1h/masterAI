import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CHANNELS, ImpactMeshDataset
from src.model import UNet


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--limit",type=int,default=8); ap.add_argument("--split"); ap.add_argument("--sample-list"); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); names=None
    if args.sample_list: names=[x.strip() for x in Path(args.sample_list).read_text(encoding="utf-8").splitlines() if x.strip()]
    ds=ImpactMeshDataset(cfg["data_root"],args.split or cfg["val_split"],cfg["input_mode"],args.limit,sample_names=names)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=UNet(CHANNELS[cfg["input_mode"]],cfg["base_channels"]).to(device)
    state=torch.load(args.checkpoint,map_location=device,weights_only=False); model.load_state_dict(state["model"]); model.eval()
    out=Path(cfg["output_dir"])/"predictions"; out.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for x,y,name in ds:
            prob=torch.sigmoid(model(x[None].to(device)))[0,0].cpu().numpy(); fig,axes=plt.subplots(1,4,figsize=(12,3))
            event_index=2 if cfg["input_mode"] in ("e2","e3") else 0
            for ax,title,a,cmap in zip(axes,["event VV","truth","probability","prediction"],[x[event_index].numpy(),y[0].numpy(),prob,prob>=.5],["gray","Blues","viridis","Blues"]): ax.imshow(a,cmap=cmap); ax.set_title(title); ax.axis("off")
            fig.suptitle(name); fig.tight_layout(); fig.savefig(out/f"{name}.png",dpi=140); plt.close(fig)


if __name__=="__main__": main()
