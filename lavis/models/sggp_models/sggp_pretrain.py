from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb as wandb

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from lavis.common.masking_generator import InteractMaskingGenerator
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.sggp_models import SggpBase

from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin
from lavis.models.med import BertForMaskedLM, UnifiedBertForMaskedLM
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from transformers import BertConfig
from detectron2 import model_zoo
from detectron2.config import instantiate

from einops import rearrange

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
@registry.register_model("SggpPretrain")
class SggpPretrain(SggpBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    ALBEF pretrain model.

    Supported model types:
        - base: ALBEF base model used for pretraining.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base_cc3m": "configs/models/sggp_pretrain_base.yaml",#就是config中"model type"的地方，决定总体模型的架构config文件路径
    }


    def __init__(
        self,
        detector,
        # text_encoder,
        unified_encoder,
        queue_size,
        embed_dim=256,
        mlm_mask_prob=0.15,
        temp=0.07,
        momentum=0.995,
        alpha=0.4,
        max_txt_len=30,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.detector = detector
        # self.visual_encoder = detector.backbone.net

        # self.text_encoder = text_encoder
        window_size = int(unified_encoder.model_cfg.image_size / unified_encoder.model_cfg.patch_size)
        self.window_size = window_size
        self.interactive_masking = InteractMaskingGenerator(window_size,unified_encoder.model_cfg.patch_size,num_masking_patches=85)
        self.unified_encoder = unified_encoder

        text_width = unified_encoder.config.hidden_size
        vision_width = detector.vision_width#768

        self.embed_dim = embed_dim

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)


        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)#todo queue 为什么对batch norm,而网络输出结果是特征norm? 因为queque的第一维是特征
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(temp * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len

        self.mlm_probability = mlm_mask_prob

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images. Default: H=224, W=224.
                - text_input (list): A list of length batch_size, each element is a string of text/caption.
                - epoch (int): The current epoch.
                - iters (int): The current iteration.
                - num_iters_per_epoch (int): The number of iterations per epoch.

        Returns:
            BlipOutput: A BlipOutput object containing loss and intermediate output. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("albef_pretrain")
            >>> images = torch.randn(4, 3, 224, 224)
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"]
            >>> samples = {"image": images, "text_input": text_input, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100}
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm', 'loss_mlm'])
        """
        image = samples["image"]
        caption = samples["text_input"]
        block_mask = samples["blocking_mask"]

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_det = []

        for i in image:
            image_det.append({"image":i})
        with torch.no_grad():
            self.detector.eval()
            detect_results = self.detector(image_det)
            interactive_mask = self.interactive_masking(detect_results)
            interactive_mask = (block_mask | interactive_mask) if interactive_mask!=None else block_mask
            #mask visualization
            # image_mask = torch.ones_like(image)
            # for b in range(image.shape[0]):
            #
            #     for i in range(interactive_mask.shape[1]):
            #         for j in range(interactive_mask.shape[2]):
            #             image_mask[b,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] \
            #                 = 1-interactive_mask[b,i,j]
            #
            # if interactive_mask!= None:
            #     plt.imshow((image[0].cpu()*image_mask[0].cpu()).permute(1,2,0))
            #     plt.show()
            # region 目标检测可视化效果
            # metadata = MetadataCatalog.get('coco_2017_train')
            # vis = Visualizer(image[0].permute(1,2,0).cpu()*255) #metadata
            # vis_pred = vis.draw_instance_predictions(detect_results[0][0]['instances'].to("cpu")).get_image()
            # plt.imshow(vis_pred)
            # plt.show()
            # endregion
        # image_embeds = self.visual_encoder.forward(image)['last_feat']
        # image_embeds = rearrange(image_embeds, 'b d w h -> b (w h) d')
        # image_embeds = self.visual_encoder.forward_features(image)


        text = self.tokenizer( #BertTokenizer:{input_ids,'token_type_ids',attention_mask}
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len, #30
            return_tensors="pt",
        ).to(self.device)
        plt.close()


        # text_output = self.text_encoder.bert(#BertModel
        #     text.input_ids,#101:cls,  102:seq
        #     attention_mask=text.attention_mask,
        #     return_dict=True,
        #     mode="text",# mode indicate the input's modelity
        # )
        # text_embeds = text_output.last_hidden_state
        # text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # unified_embeds = torch.cat((image_embeds,text_embeds),1)
        # unified_atts = torch.cat((image_atts,text.attention_mask),1)

        # MRM
        if interactive_mask is not None:
            with torch.no_grad():
                # calculate the predict label
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :, None, None]
                unnorm_images = image * std + mean  # in [0, 1]
                images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=self.unified_encoder.model_cfg.patch_size,
                                           p2=self.unified_encoder.model_cfg.patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                               ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')

        MRM_output = self.unified_encoder(  # BertModel
            input_images=image,
            patch_mask=interactive_mask if interactive_mask is not None else None,
            input_text_ids=text.input_ids,
            text_attention_mask=text.attention_mask,
            mode="multimodal",
            labels= images_patch if interactive_mask is not None else None

        )
        loss_mrm = MRM_output.loss




        # forward the positve image-text pair
        # encoder_output_pos = self.text_encoder.bert(
        #     encoder_embeds=text_embeds,
        #     attention_mask=text.attention_mask,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        #     mode="fusion",
        # )



        # MLM
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)#mlm_probability: 0.15, padding token is same probability
        input_ids, labels = self.mask(
            input_ids,
            self.unified_encoder.config.vocab_size,
            self.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )


        mlm_output = self.unified_encoder(#text_encoder:bert+classifier
            input_images=image,
            input_text_ids=input_ids,
            text_attention_mask=text.attention_mask,
            return_dict=True,
            labels=labels,
            alpha=alpha,
            mode="text"
        )
        loss_mlm = mlm_output.loss
        output={}
        output["loss_mlm"] = loss_mlm.item()
        if loss_mrm!=None:
            output["loss_mrm"] = loss_mrm.item()
        wandb.log(output)
        output["loss"] =loss_mlm+loss_mrm

        return output



    def mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False#对于padding和cls tocken,不属于mask候选的范围
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False# [cls]同上，todo[sep] 102 不false吗？

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens， -100是一定不mak的部分

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() #0.5是因为建立在所剩20%基础上的
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(
            device
        )
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    @classmethod
    def from_config(cls, cfg=None):
        detector = model_zoo.get_config(cfg.detector_config_path).model
        detector = instantiate(detector)
        detector.vision_width=cfg.vit_embed_dim
        # config_text_encoder = BertConfig.from_json_file(
        #     get_abs_path(cfg["med_config_path"])
        # )

        unified_encoder = UnifiedBertForMaskedLM.from_config(cfg,from_pretrained=True)

        # unified_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=False)
        # text_encoder = BertForMaskedLM(config_text_encoder)
        # text_encoder.fusion_layer = 6


        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        mlm_mask_prob = cfg.get("mlm_mask_prob", 0.15)
        temp = cfg.get("temp", 0.07)
        max_txt_len = cfg.get("max_txt_len", 30)
        queue_size = cfg.get("queue_size", 65536)

        model = cls(
            detector=detector,
            # text_encoder=text_encoder,
            unified_encoder=unified_encoder,
            queue_size=queue_size,
            embed_dim=embed_dim,
            mlm_mask_prob=mlm_mask_prob,
            temp=temp,
            momentum=momentum,
            alpha=alpha,
            max_txt_len=max_txt_len,
        )

        return model
