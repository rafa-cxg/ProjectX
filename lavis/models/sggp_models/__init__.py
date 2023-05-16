import datetime
import logging
import os
import time

import lavis.common.dist_utils as dist_utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.dist_utils import download_cached_file
from lavis.common.logger import MetricLogger
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.vit import interpolate_pos_embed
from transformers import BertTokenizer


class SggpBase(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        return BertTokenizer.from_pretrained("bert-base-uncased")

    def load_from_pretrained(self, url_or_filename, rename_text_keys=True):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
        #     state_dict["visual_encoder.pos_embed"], self.visual_encoder
        # )
        # if (
        #     "visual_encoder_m.pos_embed" in self.state_dict().keys()
        #     and "visual_encoder_m.pos_embed" in state_dict
        # ):
        #     state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
        #         state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
        #     )
        #
        # if rename_text_keys:
        #     for key in list(state_dict.keys()):
        #         if "bert" in key:
        #             new_key = key.replace("bert.", "")
        #             state_dict[new_key] = state_dict[key]
        #             del state_dict[key]
        #
        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg