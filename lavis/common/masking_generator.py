"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""
import itertools
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
from random import shuffle
import math
import numpy as np
import random

import torch


class InteractMaskingGenerator:
    def __init__(
            self, input_size,patch_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size
        self.patch_size = patch_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _random_mask(self, mask, max_mask_patches):
        delta = 0 # 真正新mask的patch数目
        for attempt in range(10): #认为10次一定能产生满足要求的mask操作
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)#确定左上角点坐标
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()# num_masked本次mask前，这块区域已经被mask的数目
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def _interact_mask(self, mask, max_mask_patches,union_box):
        delta = 0  # 真正新mask的patch数目

        h =  abs(union_box[3] - union_box[1])
        w = abs(union_box[2] - union_box[0])
        x = min(union_box[2],union_box[0])#x
        y = min(union_box[1], union_box[3])

        num_masked = mask[x: x+w,y: y+h].sum()
        max_mask_patches = 12000
        if 0 < w*h  - num_masked <= max_mask_patches:
            for i in range(x, x + w):
                for j in  range(y, y + h):
                    if mask[i, j] == 0:
                        mask[i, j] = 1
                        delta += 1
        return mask
    
    

    def __call__(self, detection_results):

        masks=[]
        for detection_result in detection_results[0]:#include batches
            mask = np.zeros(shape=self.get_shape(), dtype=np.long)
            num_boxes = len(detection_result['instances'])

            if num_boxes == 0 or num_boxes == 1:
                return None
            else:
                combine_boxes = list(itertools.combinations(detection_result['instances'].pred_boxes,2))
                # shuffle(combine_boxes)
                for boxes in combine_boxes:

                    box1 = boxes[0]
                    box2 = boxes[1]
                    xmin = math.floor(int((max(box1[0], box2[0]))/self.patch_size))
                    xmax = math.ceil(int((min(box1[2], box2[2]))/self.patch_size))
                    ymin = math.floor(int((max(box1[1], box2[1]))/self.patch_size))
                    ymax = math.ceil(int((min(box1[3], box2[3]))/self.patch_size))
                    # if (xmax-xmin)(ymax-ymin) >0: todo 当前策略：不必两两相交
                    union_box = (xmin,ymin,xmax,ymax)
                    mask = self._interact_mask(mask, self.max_num_patches, union_box)

                masks.append(torch.as_tensor(mask,device=box1.device))
        return torch.stack(masks,0)

class BlockMaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0 # 真正新mask的patch数目
        for attempt in range(10): #认为10次一定能产生满足要求的mask操作
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)#确定左上角点坐标
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches: #要mask固定数目的才不再mask
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
