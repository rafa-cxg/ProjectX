"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

import matplotlib.pyplot as plt

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class ImageTextPairDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        #########################add by cxg, for cc only################################
        # if 'training' in ann["image"]:
        #     ann["image"]=self.vis_root+'/'+ann["image"].strip('export/share/datasets/vision/conceptual_captions/DownloadConceptualCaptions/training')+'.jpg'
        # if 'validicting' in ann["image"]:
        #     ann["image"].strip('validicting')
        try:
            image_path = os.path.join(self.vis_root, ann["image"])
        except:
            print(ann["image"])
        image = Image.open(image_path).convert("RGB")
        # plt.imshow(image)
        # plt.show()
        image,blocking_mask = self.vis_processor(image)#todo RandomHorizontalFlip(p=0.5) 应该去掉，否则影响SGG
        # plt.imshow(image.permute(1,2,0))
        # plt.show()
        caption = self.text_processor(ann["caption"])

        return {"image": image, "text_input": caption, "blocking_mask": blocking_mask}
