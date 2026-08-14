import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import rasterio
import zarr
from zarr.storage import ZipStore


def main():
    root=Path("data/raw/val"); sums=defaultdict(lambda:np.zeros(8)); counts=defaultdict(lambda:np.zeros(8)); pixels=defaultdict(int)
    masks=list((root/"MASK").rglob("*_annotation_flood.tif"))
    for mask_path in masks:
        name=mask_path.name.removesuffix("_annotation_flood.tif"); sar_path=next(iter((root/"S1RTC").rglob(name+"_S1RTC.zarr.zip")),None)
        if sar_path is None: continue
        with rasterio.open(mask_path) as src: mask=src.read(1)
        with ZipStore(sar_path,mode="r") as store: sar=np.asarray(zarr.open_group(store,mode="r")["bands"][:],dtype=np.float32).reshape(8,256,256)
        for cls in (0,1,2):
            q=mask==cls; pixels[cls]+=int(q.sum())
            for channel in range(8):
                values=sar[channel][q]; finite=np.isfinite(values); sums[cls][channel]+=values[finite].sum(); counts[cls][channel]+=finite.sum()
    labels={0:"background",1:"permanent_water",2:"flood"}; report={}
    for cls in (0,1,2): report[str(cls)]={"semantic":labels[cls],"pixels":pixels[cls],"mean_db":np.round((sums[cls]/counts[cls]).reshape(4,2),3).tolist(),"rows":["pre_month","pre_event","event","post_event"],"columns":["VV","VH"]}
    out=Path("outputs/label_semantics.json"); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(out)


if __name__=="__main__": main()
