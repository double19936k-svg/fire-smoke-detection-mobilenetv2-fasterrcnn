"""Plot training and validation history for the project."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_training_records(path):
    if not path.is_file():
        return None

    data = load_json(path)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("epochs") or data.get("history") or data.get("records")
    else:
        raise ValueError("training_log.json 格式错误，应为记录列表。")

    if not isinstance(records, list) or not records:
        raise ValueError("training_log.json 中没有 epoch 记录。")
    return [record for record in records if isinstance(record, dict)]


def value(record, *keys):
    for key in keys:
        if key in record and record[key] is not None:
            return float(record[key])
    return None


def plot_series(ax, x, series, title, ylabel, colors):
    has_data = False
    for label, values, color in series:
        if any(item is not None for item in values):
            valid = [item if item is not None else float("nan") for item in values]
            ax.plot(x, valid, marker="o", linewidth=2.2, label=label, color=color)
            has_data = True

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if has_data:
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5,
            0.5,
            "training_log.json not found\nRun train.py to record history",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="#5D6B82",
        )


def plot_metrics(ax, records, evaluation_results):
    labels = ["mAP50", "mAP50-95", "precision", "recall"]
    colors = ["#F97316", "#14B8A6", "#6366F1", "#EAB308"]

    if records:
        epochs = [int(record["epoch"]) for record in records if "epoch" in record]
        series = []
        for label, color in zip(labels, colors):
            values = [value(record, label) for record in records]
            series.append((label, values, color))
        plot_series(ax, epochs, series, "Detection Metrics", "Score", colors)
        ax.set_ylim(0, 1)
        return

    ax.set_title("Detection Metrics", fontsize=16, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    if evaluation_results is None:
        ax.text(
            0.5,
            0.5,
            "No evaluation_results.json found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="#5D6B82",
        )
        return

    result_values = [
        value(evaluation_results, "mAP50", "mAP@0.5"),
        value(evaluation_results, "mAP50_95", "mAP@0.5:0.95"),
        value(evaluation_results, "precision"),
        value(evaluation_results, "recall"),
    ]
    bars = ax.bar(labels, [item if item is not None else 0 for item in result_values], color=colors)
    for bar, item in zip(bars, result_values):
        label = "N/A" if item is None else f"{item:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            (item or 0) + 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Detection Metrics", fontsize=16, fontweight="bold")


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"
    parser = argparse.ArgumentParser(description="Plot training_curve.png.")
    parser.add_argument(
        "--log",
        type=Path,
        default=results_dir / "training_log.json",
        help="training_log.json 路径。",
    )
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=results_dir / "evaluation_results.json",
        help="evaluation_results.json 路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir / "training_curve.png",
        help="输出 PNG 路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_training_records(args.log)
    evaluation_results = (
        load_json(args.evaluation_results)
        if args.evaluation_results.is_file()
        else None
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    fig.suptitle(
        "MobileNetV2 + Faster R-CNN Training and Evaluation",
        fontsize=20,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.32, wspace=0.22, top=0.90)

    if records:
        epochs = [int(record["epoch"]) for record in records if "epoch" in record]
        plot_series(
            axes[0, 0],
            epochs,
            [
                ("train_loss", [value(record, "train_loss") for record in records], "#F97316"),
                ("val_loss", [value(record, "val_loss") for record in records], "#DC2626"),
            ],
            "Loss Curves",
            "Loss",
            ["#F97316", "#DC2626"],
        )
        plot_series(
            axes[0, 1],
            epochs,
            [
                ("cls_loss", [value(record, "cls_loss") for record in records], "#6366F1"),
                ("box_loss", [value(record, "box_loss") for record in records], "#8B5CF6"),
            ],
            "Classification / Regression Loss",
            "Loss",
            ["#6366F1", "#8B5CF6"],
        )
        plot_series(
            axes[1, 0],
            epochs,
            [("learning_rate", [value(record, "learning_rate") for record in records], "#14B8A6")],
            "Learning Rate",
            "Learning rate",
            ["#14B8A6"],
        )
    else:
        for ax, title in zip(
            axes.flat[:3],
            ["Loss Curves", "Classification / Regression Loss", "Learning Rate"],
        ):
            ax.set_title(title, fontsize=16, fontweight="bold")
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "training_log.json not found\nRun train.py to record history",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                color="#5D6B82",
            )

    plot_metrics(axes[1, 1], records, evaluation_results)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    if records:
        print(f"已生成训练曲线：{args.output.resolve()}")
    else:
        print(
            f"已生成提示版图片：{args.output.resolve()}；"
            "重新运行 train.py 后可生成真实曲线。"
        )


if __name__ == "__main__":
    main()
