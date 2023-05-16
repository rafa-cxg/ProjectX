"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import logging

import numpy as np
import torch
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from evaluation import evaluate
from lavis.tasks.base_task import BaseTask


@registry.register_task("scene_graph_generation")
class SGGTask(BaseTask):
    def __init__(self):
        super().__init__()

    def valid_step(self, model, samples):
        results = []

        outputs = model.predict(samples)

        predictions = outputs["predictions"]
        instance_gts = outputs["instance_gts"]
        indices = samples["index"]
        # for pred, tgt, index in zip(predictions, instance_gts, indices):
        #     pred = pred.max(-1)[1].cpu().tolist()
        #     relation_targets = tgt.get("relation_tuple")[:, -1].cpu().tolist()
        #     if isinstance(index, torch.Tensor):
        #         index = index.item()

        results.append(
            {
                self.inst_id_key: indices,
                "prediction": predictions,
                "relation_target": instance_gts,
            }
        )

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):

        # eval_result_file = self.save_result(
        #     result=val_result,
        #     result_dir=registry.get_path("result_dir"),
        #     filename="{}_epoch{}".format(split_name, epoch),
        #     remove_duplicate=self.inst_id_key,
        # )

        metrics = self._report_metrics(
            eval_result=val_result, split_name=split_name
        )


        return metrics

    @main_process #only rank=0 process wil caculate
    def _report_metrics(self, eval_result, split_name):


        predictions = [res.get("prediction") for res in eval_result]
        targets = [res.get("relation_target") for res in eval_result]

        # accuracy = (targets == predictions).sum() / targets.shape[0]
        #
        result=evaluate(predictions,targets,iou_types=("relations",))
        metrics={"agg_metrics":result[0],"detail_metrics":result[1]}#agg_metrics选的是mrecall
        # for result in result[0]:
        #     metrics.update(result)
        # log_stats = {split_name: {k: v for k, v in metrics.items()}}



        # logging.info(metrics)
        return metrics

