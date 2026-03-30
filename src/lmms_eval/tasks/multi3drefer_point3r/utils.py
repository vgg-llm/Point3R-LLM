import re
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from scipy.optimize import linear_sum_assignment
from loguru import logger as eval_logger
from lmms_eval.tasks.threedod.utils import EulerDepthInstance3DBoxes

with open(Path(__file__).parent / "multi3drefer_point3r.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]


def multi3drefer_doc_to_visual(doc):
    return [[]]


def multi3drefer_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["conversations"][0]["value"].replace("<image>", "")
    return question


def _compute_iou_matrix(pred_bboxes, gt_bboxes):
    """Compute N x M IoU matrix between predicted and GT bboxes."""
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return np.zeros((len(pred_bboxes), len(gt_bboxes)))
    pred_boxes = EulerDepthInstance3DBoxes(torch.tensor(pred_bboxes, dtype=torch.float32), convention="ZXY")
    gt_boxes = EulerDepthInstance3DBoxes(torch.tensor(gt_bboxes, dtype=torch.float32), convention="ZXY")
    iou_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i in range(len(pred_bboxes)):
        for j in range(len(gt_bboxes)):
            iou_matrix[i, j] = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(torch.tensor([pred_bboxes[i]], dtype=torch.float32), convention="ZXY"),
                EulerDepthInstance3DBoxes(torch.tensor([gt_bboxes[j]], dtype=torch.float32), convention="ZXY"),
            ).item()
    return iou_matrix


def _f1_from_iou_matrix(iou_matrix, threshold):
    """Compute F1 score at a given IoU threshold using Hungarian matching."""
    n_pred, n_gt = iou_matrix.shape
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    # Hungarian matching on IoU matrix
    row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)
    tp = sum(1 for r, c in zip(row_idx, col_idx) if iou_matrix[r, c] >= threshold)

    precision = tp / n_pred
    recall = tp / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def multi3drefer_process_results(doc, results):
    raw = results[0].strip('\n').strip("```").strip("json").strip("\n")
    gt_bboxes_cam = doc.get("gt_bboxes_cam", [])

    pred_bboxes = None
    # Try to parse the JSON response
    for line in raw.split("\n"):
        if "bbox_3d" in line:
            try:
                pred_dict = eval(line.strip())
                if "bbox_3d" in pred_dict and isinstance(pred_dict["bbox_3d"], list):
                    pred_bboxes = pred_dict["bbox_3d"]
            except Exception:
                pass
            break
    # Fallback: try parsing the whole response as JSON
    if pred_bboxes is None:
        try:
            pred_dict = json.loads(raw)
            if "bbox_3d" in pred_dict:
                pred_bboxes = pred_dict["bbox_3d"]
        except Exception:
            pass

    if pred_bboxes is None:
        pred_bboxes = []

    # Validate: must be list of 9-element lists
    valid_preds = []
    for b in pred_bboxes:
        if isinstance(b, (list, tuple)) and len(b) == 9:
            valid_preds.append(list(b))
        else:
            eval_logger.warning(f"Skipping invalid pred bbox: {b}")

    f1_25, f1_50 = 0.0, 0.0
    try:
        iou_matrix = _compute_iou_matrix(valid_preds, gt_bboxes_cam)
        f1_25 = _f1_from_iou_matrix(iou_matrix, 0.25)
        f1_50 = _f1_from_iou_matrix(iou_matrix, 0.50)
    except Exception as e:
        eval_logger.error(f"Error computing IoU: {e}")

    ret = {
        'f1_25': f1_25,
        'f1_50': f1_50,
        'question_type': doc.get('metadata', {}).get('question_type', 'unknown'),
        'ann_id': doc.get('metadata', {}).get('ann_id', -1),
        'pred_bboxes': valid_preds,
        'gt_bboxes_cam': gt_bboxes_cam,
    }
    return {"multi3drefer_score": ret}


def multi3drefer_aggregate_results(results):
    results = pd.DataFrame(results)

    output = {}
    output["f1_25"] = results["f1_25"].mean() * 100
    output["f1_50"] = results["f1_50"].mean() * 100

    # Per-type breakdown
    for qtype in results["question_type"].unique():
        mask = results["question_type"] == qtype
        output[f"f1_25_{qtype}"] = results.loc[mask, "f1_25"].mean() * 100
        output[f"f1_50_{qtype}"] = results.loc[mask, "f1_50"].mean() * 100

    eval_logger.info(f"Multi3DRefer results: {output}")
    return output["f1_25"]
