import json
import platform
from pathlib import Path
import numpy
import rasterio
import torch
import zarr


info={"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"numpy":numpy.__version__,"zarr":zarr.__version__,"rasterio":rasterio.__version__}
out=Path("outputs/environment.json"); out.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8"); print(out,info)
