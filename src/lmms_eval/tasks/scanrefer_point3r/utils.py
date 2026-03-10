import re
import os
import torch
import pandas as pd
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from loguru import logger as eval_logger
from lmms_eval.tasks.threedod.utils import EulerDepthInstance3DBoxes

with open(Path(__file__).parent / "scanrefer_point3r.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]


def scanrefer_doc_to_visual(doc):
    image_files = doc.get("images", [])
    images = [
        Image.open(
            os.path.join(media_dir, image_file)
        ).convert("RGB")
        for image_file in image_files
    ]
    return [images]


def scanrefer_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["prompt"]
    return prompt


def scanrefer_process_results(doc, results):
    lines = results[0].strip('\n').strip("```").strip("json").strip("\n").split("\n")
    gt_bbox_cam = doc["gt_bbox_cam"]
    pred_dict = None
    for line in lines:
        if "bbox_3d" in line:
            try:
                pred_dict = eval(line.strip())
            except Exception as e:
                eval_logger.error(f"Error parsing bbox_3d: {line.strip()}")
            break

    iou = 0
    pred_bbox = None
    if pred_dict is not None:
        try:
            assert "bbox_3d" in pred_dict and isinstance(pred_dict["bbox_3d"], list) and len(pred_dict["bbox_3d"]) == 9, \
                "Invalid bbox_3d format"

            pred_bbox = pred_dict["bbox_3d"]
            # IoU computed directly in camera space (invariant under rigid transforms)
            iou = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(torch.tensor([pred_bbox]), convention="ZXY"),
                EulerDepthInstance3DBoxes(torch.tensor([gt_bbox_cam]), convention="ZXY")
            ).item()
        except Exception as e:
            eval_logger.error(f"Error parsing pred_dict: {pred_dict} with error: {e}")

    ret = {
        'iou': iou,
        'pred_bbox': pred_bbox,
        'gt_bbox_cam': gt_bbox_cam,
    }
    return {"scanrefer_score": ret}


def scanrefer_aggregate_results(results):
    results = pd.DataFrame(results)

    output = {}
    output["iou25"] = (results["iou"] >= 0.25).mean() * 100
    output["iou50"] = (results["iou"] >= 0.50).mean() * 100

    eval_logger.info(f"Scanrefer results: {output}")
    return output["iou25"]
