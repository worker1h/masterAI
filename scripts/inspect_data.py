import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
import zarr
from zarr.storage import ZipStore
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",default="data/raw"); ap.add_argument("--split",default="val"); ap.add_argument("--limit",type=int,default=20); ap.add_argument("--output",default="outputs/data_audit.json"); args=ap.parse_args()
    root=Path(args.data_root)/args.split
    masks=[]; seen=set()
    for path in sorted((root/"MASK").rglob("*_annotation_flood.tif")):
        event=path.name.split("_",1)[0]
        if event not in seen: masks.append(path); seen.add(event)
        if len(masks)>=args.limit: break
    report=[]; figure_dir=Path("outputs/data_audit_figures"); figure_dir.mkdir(parents=True,exist_ok=True)
    for mask_path in masks:
        name=mask_path.name.removesuffix("_annotation_flood.tif"); item={"sample":name,"modalities":{}}
        paths={"MASK":mask_path,"DEM":next(iter((root/"DEM").rglob(name+"_DEM.tif")),None),"S1RTC":next(iter((root/"S1RTC").rglob(name+"_S1RTC.zarr.zip")),None)}
        for mod,path in paths.items():
            if path is None: item["modalities"][mod]={"missing":True}; continue
            if mod=="S1RTC":
                with ZipStore(path,mode="r") as store:
                    group=zarr.open_group(store,mode="r"); key="bands"; a=np.asarray(group[key][...]); times=np.asarray(group["time"][:]).astype("datetime64[ns]").astype(str).tolist(); meta={"variable":key,"arrays":list(group.keys()),"times":times,"bands":np.asarray(group["band"][:]).astype(str).tolist()}
            else:
                with rasterio.open(path) as src: a=src.read(); meta={"crs":str(src.crs),"transform":list(src.transform)[:6]}
            finite=np.isfinite(a); item["modalities"][mod]={**meta,"shape":list(a.shape),"dtype":str(a.dtype),"min":float(np.nanmin(a)),"max":float(np.nanmax(a)),"missing_fraction":float(1-finite.mean())}
        if all(not item["modalities"][m].get("missing",False) for m in ("S1RTC","DEM","MASK")):
            s1_path=paths["S1RTC"]
            with ZipStore(s1_path,mode="r") as store: sar=np.asarray(zarr.open_group(store,mode="r")["bands"][:])
            with rasterio.open(paths["DEM"]) as src: dem=src.read(1)
            with rasterio.open(paths["MASK"]) as src: mask=src.read(1)
            fig,axes=plt.subplots(2,3,figsize=(10,7)); titles=["pre-month VV","pre-event VV","event VV","post-event VV","DEM","MASK"]
            arrays=[sar[0,0],sar[1,0],sar[2,0],sar[3,0],dem,mask]
            for ax,title,array in zip(axes.flat,titles,arrays): ax.imshow(array,cmap="gray" if title!="MASK" else "Blues"); ax.set_title(title); ax.axis("off")
            fig.suptitle(name); fig.tight_layout(); fig.savefig(figure_dir/f"{name}.png",dpi=130); plt.close(fig)
        report.append(item)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(out, len(report))


if __name__ == "__main__": main()
