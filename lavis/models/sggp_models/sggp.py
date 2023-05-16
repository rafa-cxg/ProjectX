import json
import os
import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import BertConfig

from detectron2 import model_zoo
from detectron2.config import instantiate
from lavis.common.registry import registry
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY as detrectron_registry
from lavis.models.sggp_models import SggpBase
from lavis.models.sggp_models.sggp_outputs import (
    SggpIntermediateOutput,
    SggpOutputWithLogits,
)
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.med import XBertEncoder, BertLMHeadModel, BertForMaskedLM
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from detectron2.modeling.backbone import SimpleFeaturePyramid, ViT
from detectron2.config import LazyConfig, instantiate
from detectron2.structures import Instances,Boxes
from detectron2.utils.events import EventStorage
from lavis.common.utils import get_abs_path
# from lavis.datasets.datasets.vg_sgg_datasets import BoxList

@registry.register_model("SggpFinetune")
class SggpFinetune(SggpBase):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_cc3m": "configs/models/sggp_cc3m.yaml",
    }

    def __init__(
        self,
        # image_encoder,
        text_encoder,
        detector,
        decoder,
        num_classes,
        momentum=0.995,
        alpha=0.4,
        use_distill=True,
        max_txt_len=40,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.max_txt_len = max_txt_len
        self.use_distill = use_distill
        self.detector = detector
        self.decoder = decoder
        # self.visual_encoder = image_encoder
        self.text_encoder = text_encoder
        cfg_file=OmegaConf.load("lavis/configs/datasets/vg/defaults_sgg.yaml")
        vocab_file=json.load(open(os.path.join("cache",cfg_file.datasets.visual_genome.build_info.annotations.train.storage[1])))
        self.idx_to_label =vocab_file['idx_to_label']
        hidden_size = text_encoder.config.hidden_size
        # self.text_encoder.encoder.gradient_checkpointing=True
        # self.decoder.bert.encoder.gradient_checkpointing = True
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            warnings.warn(
                f"Found num_classes=0, initializing {type(self)} without classifier."
            )



    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):

        # construct detrectron2 data structure:
        instance_gts=[]
        for label in samples["labels"]:
            instance_gts.append(conver_boxlist_to_instance(label))
        detrectron_data=[]
        relation_targets=[]

        # strack image list to bs tensor
        # image=torch.stack(samples["image"], 0)
        # image_embeds = self.detector.backbone.net(image)['last_feat']
        for i in range(len(samples["index"])):
            relation_targets.append(instance_gts[i].get("relation_tuple")[:, -1].to(self.device))
            meta = {}
            for key, value in samples.items():
                if isinstance(value,int):
                    continue
                meta[key]=value[i]
                meta["instances"]=instance_gts[i]
            # relation_targets=torch.cat(relation_targets,0).to(self.device)
            detrectron_data.append(meta)

        # get detector's result
        with EventStorage() as storage:
             loss_dict,vit_output=self.detector(detrectron_data)
            # del proposals, detrectron_data
        samples["relation_tuple"] = []
        for instance_gt in instance_gts:
            relation_sentance = []
            label=instance_gt.get("labels")
            for triplet in instance_gt.get("relation_tuple"):
                temp=self.idx_to_label[str(int(label[int(triplet[0])]))] + ' '+'[MASK]'+' '+ self.idx_to_label[str(int(label[int(triplet[1])]))]
                relation_sentance.append(temp)
            samples["relation_tuple"].append(relation_sentance)
        sentences = []
        for relation_tuples in samples["relation_tuple"]:
            temp=[]
            for tuple in relation_tuples:

                sentence = self.tokenizer(
                    tuple,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)
                temp.append(sentence)
            sentences.append(temp)

        relation_loss = []
        predictions = []
        for tokenized_texts,batch_relation_targets in zip(sentences,relation_targets):

            prediction=[]
            for batch_vit_output,tokenized_text,relation_target in zip(vit_output,tokenized_texts,batch_relation_targets):
                text_encoder_output = self.text_encoder.forward_text(
                    tokenized_text)
                labels=relation_target.view(1,-1) if is_train else None
                decoder_output=self.decoder(inputs_embeds=text_encoder_output.last_hidden_state[:, :, :],
                                            encoder_hidden_states=batch_vit_output.permute(1,2,0).unsqueeze(0).flatten(start_dim=1,end_dim=2),
                                            labels=labels,mode="multimodal")# “0” 代表选择[cls] tocken
                if is_train:
                    relation_loss.append(decoder_output.loss)
                else:

                    prediction.append(decoder_output.logits[:,2,:].unsqueeze(1))#2 is [mask] position
            if is_train is False:
                prediction = torch.cat(prediction, 0).squeeze().cpu()
                predictions.append(prediction)
            # relation_loss.append(F.cross_entropy(predictions, relation_target))


        # del samples, image_embeds, instance_gts
        if is_train:
            relation_loss = sum(relation_loss) / len(samples["image"])
            loss_dict["relation_loss"] = relation_loss
            return {"loss": sum(loss_dict.values())}
            # return SggpOutputWithLogits(
            #     loss= sum(loss_dict.values()),
                # intermediate_output=SggpIntermediateOutput(
                #     image_embeds=image_embeds,
                #
                # ),
            # )
        else:
            return {"predictions": predictions, "instance_gts":instance_gts}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        detector=model_zoo.get_config(cfg.detector_config_path).model
        # image_encoder = VisionTransformerEncoder.from_config(cfg)
        detector=instantiate(detector)
        # detector=detrectron_registry.get("GeneralizedRCNN").from_config(cfg)
        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.from_config(cfg)
        decoder_cfg=BertConfig.from_json_file(get_abs_path(cfg.med_config_path))
        decoder_cfg.vocab_size=51
        decoder = BertForMaskedLM(decoder_cfg)

        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 40)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            # image_encoder=image_encoder,
            text_encoder=text_encoder,
            detector=detector,
            decoder=decoder,
            use_distill=use_distill,
            alpha=alpha,
            num_classes=num_classes,
            momentum=momentum,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model
'''
creator: cxg
for converting pysgg structure "boxlist" to detrectron's "instances"
'''
def conver_boxlist_to_instance(boxlist):
    structure=Instances(boxlist.size)
    gt_boxes=Boxes(boxlist.bbox)
    structure.set("gt_boxes",gt_boxes)

    for field in boxlist.fields():
        structure.set(field,boxlist.get_field(field))
    structure.set("gt_classes", boxlist.get_field("labels"))
    structure.set("relation_tuple", boxlist.get_field("relation_tuple"))
    return structure


