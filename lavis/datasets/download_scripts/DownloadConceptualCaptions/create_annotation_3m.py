import os
import json

import pandas as pd
from tqdm import tqdm
from lavis.common.utils import get_abs_path, get_cache_path

cc3m = pd.read_csv("downloaded_cc3m_report.tsv.gz", compression="gzip", sep="\t", names=["caption", "path", "dataset", "mimetype", "size", "status", "url"])
cc3m.iloc[15]
len(cc3m)
cnt = 0

valid_records = []

for i, path in tqdm(enumerate(cc3m.path.unique()), total=len(cc3m.path.unique())):
    path = str(path)
    if os.path.exists(path):
        record = cc3m.iloc[i]
        if isinstance(record["path"],str) and  os.path.exists(record["path"]):#不知道为什么nan不能被path.unique排除？
            valid_records.append({"image": record["path"], "caption": record["caption"]})
            cnt += 1
        # else:
        #     print('f')

print("Found {} valid records".format(cnt))

len(valid_records)

valid_records[1]

from omegaconf import OmegaConf


config_path = get_abs_path("configs/datasets/conceptual_caption/defaults_3m.yaml")

# ann_path = OmegaConf.load(
#     config_path
# ).datasets.conceptual_caption_3m.build_info.annotations.train.storage[0]
#
# ann_path = get_cache_path(ann_path)
ann_path='/datasets/conceptual_captions/cache/conceptual_caption/annotations/cc3m_train.json'

if os.path.exists(ann_path):
    # abort
    print("{} already exists".format(ann_path))
else:
    # Save the valid records to a json file
    with open(ann_path, "w") as f:
        f.write(json.dumps(valid_records))