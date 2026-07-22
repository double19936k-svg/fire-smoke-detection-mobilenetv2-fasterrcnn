"""Evaluate the trained MobileNetV2 + Faster R-CNN model on the val split.

The project uses YOLO-format label files, while Faster R-CNN uses class ids
starting at 1.  FireDataset already performs that conversion, so this script
reuses it unchanged and computes class-wise AP for fire (1) and smoke (2).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from dataset import FireDataset, collate_fn


CLASS_NAMES = {1: "fire", 2: "smoke"}
IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def build_model():
    """Build the same architecture as model.py without downloading weights.

    The trained state_dict replaces every parameter, so ImageNet initialization
    is unnecessary during evaluation.  Omitting it also makes evaluation work
    offline and does not alter the architecture used by the training script.
    """

    backbone = torchvision.models.mobilenet_v2(weights=None).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2,
    )

    return FasterRCNN(
        backbone,
        num_classes=3,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )


def load_weights(model, weights_path, device):
    """Load the state_dict saved by train.py, with common checkpoint support."""

    try:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "权重文件不是 state_dict；期望 train.py 保存的模型参数字典。"
        )

    # Also accept checkpoints saved through DataParallel.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module.") :]: value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU for two [N, 4] and [M, 4] box tensors."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    area1_wh = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0)
    area2_wh = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0)
    area1 = area1_wh[:, 0] * area1_wh[:, 1]
    area2 = area2_wh[:, 0] * area2_wh[:, 1]
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / union.clamp(min=torch.finfo(intersection.dtype).eps)


def average_precision(predictions, ground_truths, iou_threshold):
    """Compute AP using all predictions and 101-point precision interpolation."""

    number_of_ground_truths = sum(len(boxes) for boxes in ground_truths.values())
    if number_of_ground_truths == 0:
        return None

    predictions = sorted(predictions, key=lambda item: item[0], reverse=True)
    matched = {
        image_id: torch.zeros(len(boxes), dtype=torch.bool)
        for image_id, boxes in ground_truths.items()
    }

    true_positives = torch.zeros(len(predictions), dtype=torch.float64)
    false_positives = torch.zeros(len(predictions), dtype=torch.float64)

    for index, (_, image_id, predicted_box) in enumerate(predictions):
        image_ground_truths = ground_truths.get(image_id)
        if image_ground_truths is None or len(image_ground_truths) == 0:
            false_positives[index] = 1
            continue

        ious = box_iou(predicted_box.unsqueeze(0), image_ground_truths)[0]
        best_iou, best_index = ious.max(dim=0)
        if (
            best_iou.item() >= iou_threshold
            and not matched[image_id][best_index]
        ):
            matched[image_id][best_index] = True
            true_positives[index] = 1
        else:
            false_positives[index] = 1

    if len(predictions) == 0:
        return 0.0

    cumulative_tp = torch.cumsum(true_positives, dim=0)
    cumulative_fp = torch.cumsum(false_positives, dim=0)
    recall = cumulative_tp / number_of_ground_truths
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp(min=1e-12)

    # COCO-style 101-point interpolation: AP is the mean of the maximum
    # precision at recall levels 0.00, 0.01, ..., 1.00.
    recall_levels = torch.linspace(0, 1, 101, dtype=torch.float64)
    interpolated_precision = []
    for recall_level in recall_levels:
        valid = precision[recall >= recall_level]
        interpolated_precision.append(
            valid.max().item() if len(valid) else 0.0
        )

    return float(sum(interpolated_precision) / len(interpolated_precision))


def compute_map(predictions_by_class, ground_truths_by_class):
    """Compute AP at IoU 0.50 and COCO mAP over 0.50:0.95."""

    ap_by_class = {}
    valid_classes = []

    for class_id, class_name in CLASS_NAMES.items():
        ground_truths = ground_truths_by_class[class_id]
        if sum(len(boxes) for boxes in ground_truths.values()) > 0:
            valid_classes.append(class_id)

        ap_by_class[class_name] = {}
        for threshold in IOU_THRESHOLDS:
            ap_by_class[class_name][f"{threshold:.2f}"] = average_precision(
                predictions_by_class[class_id],
                ground_truths,
                threshold,
            )

    if not valid_classes:
        raise RuntimeError("验证集没有可用的 fire/smoke 标注框，无法计算 mAP。")

    def mean_defined(values):
        values = [value for value in values if value is not None]
        return float(sum(values) / len(values)) if values else 0.0

    map50 = mean_defined(
        ap_by_class[CLASS_NAMES[class_id]]["0.50"]
        for class_id in valid_classes
    )
    map50_95 = mean_defined(
        ap_by_class[CLASS_NAMES[class_id]][f"{threshold:.2f}"]
        for class_id in valid_classes
        for threshold in IOU_THRESHOLDS
    )

    return {
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map50_95,
        "per_class_AP": ap_by_class,
        "classes_used_for_mean": [CLASS_NAMES[class_id] for class_id in valid_classes],
    }


def evaluate_model(model, loader, device, score_threshold=0.0):
    """Run inference and collect predictions/ground truths by class and image."""

    predictions_by_class = {class_id: [] for class_id in CLASS_NAMES}
    ground_truths_by_class = {class_id: {} for class_id in CLASS_NAMES}
    total_images = 0
    total_predictions = 0
    total_inference_time = 0.0

    model.eval()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images_on_device = [image.to(device) for image in images]

            start_time = time.perf_counter()
            outputs = model(images_on_device)
            total_inference_time += time.perf_counter() - start_time

            for image_offset, (target, output) in enumerate(zip(targets, outputs)):
                image_id = total_images + image_offset

                for class_id in CLASS_NAMES:
                    target_mask = target["labels"] == class_id
                    ground_truths_by_class[class_id][image_id] = target[
                        "boxes"
                    ][target_mask].cpu()

                boxes = output["boxes"].detach().cpu()
                labels = output["labels"].detach().cpu()
                scores = output["scores"].detach().cpu()
                keep = scores >= score_threshold
                boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

                for box, label, score in zip(boxes, labels, scores):
                    class_id = int(label.item())
                    if class_id in CLASS_NAMES:
                        predictions_by_class[class_id].append(
                            (float(score.item()), image_id, box)
                        )
                        total_predictions += 1

            total_images += len(images)

            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"已评估 {total_images}/{len(loader.dataset)} 张图像，"
                    f"累计预测框 {total_predictions}"
                )

    metrics = compute_map(predictions_by_class, ground_truths_by_class)
    metrics["images"] = total_images
    metrics["predictions"] = total_predictions
    metrics["avg_inference_time_ms_per_image"] = (
        total_inference_time / total_images * 1000 if total_images else 0.0
    )
    metrics["ground_truth_boxes"] = {
        CLASS_NAMES[class_id]: sum(
            len(boxes) for boxes in ground_truths_by_class[class_id].values()
        )
        for class_id in CLASS_NAMES
    }
    return metrics


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    data_root = project_dir

    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on val.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_root,
        help="包含 train/val/test 文件夹的数据集根目录。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_dir / "weights" / "mobilenetv2_fasterrcnn.pth",
        help="训练得到的 .pth 权重文件。",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="评估前过滤预测框的分数阈值；AP通常应保留默认0.0。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "results" / "evaluation_results.json",
        help="保存最终指标的 JSON 文件路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前 PyTorch 检测不到 CUDA。")

    device = torch.device(args.device)
    val_dir = args.data_root / "val"
    if not (val_dir / "images").is_dir() or not (val_dir / "labels").is_dir():
        raise FileNotFoundError(
            f"验证集目录不完整，期望存在：{val_dir / 'images'} 和 {val_dir / 'labels'}"
        )
    if not args.weights.is_file():
        raise FileNotFoundError(f"找不到权重文件：{args.weights}")

    print(f"设备: {device}")
    print(f"验证集: {val_dir}")
    print(f"权重: {args.weights}")

    dataset = FireDataset(str(val_dir))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"验证集图像数: {len(dataset)}")

    model = build_model().to(device)
    load_weights(model, args.weights, device)
    print("模型权重加载完成。")

    metrics = evaluate_model(model, loader, device, args.score_threshold)

    print("\n===== Evaluation Results =====")
    print(f"mAP@0.5       : {metrics['mAP@0.5']:.6f}")
    print(f"mAP@0.5:0.95  : {metrics['mAP@0.5:0.95']:.6f}")
    print("Per-class AP@0.5:")
    for class_name in CLASS_NAMES.values():
        ap50 = metrics["per_class_AP"][class_name]["0.50"]
        print(f"  {class_name:5s}: {ap50:.6f}" if ap50 is not None else f"  {class_name:5s}: N/A")
    print(
        "平均单张推理时间: "
        f"{metrics['avg_inference_time_ms_per_image']:.2f} ms"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f"\n完整指标已保存到: {args.output.resolve()}")


if __name__ == "__main__":
    main()
