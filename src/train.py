"""Train MobileNetV2 + Faster R-CNN and record metrics."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FireDataset, collate_fn
from evaluate import evaluate_model
from model import get_model


# =========================
# 路径配置
# =========================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_DIR / "data"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

WEIGHTS_PATH = PROJECT_DIR / "weights" / "mobilenetv2_fasterrcnn.pth"
TRAIN_LOG_PATH = PROJECT_DIR / "results" / "training_log.json"


# =========================
# 参数
# =========================

EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 0



def save_log(log):

    with open(
        TRAIN_LOG_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            log,
            f,
            ensure_ascii=False,
            indent=2
        )



def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print("使用设备:", device)


    # =========================
    # 数据
    # =========================

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"训练集不存在:{TRAIN_DIR}"
        )


    train_dataset = FireDataset(
        str(TRAIN_DIR)
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
        if device.type=="cuda"
        else False
    )


    val_dataset = FireDataset(
        str(VAL_DIR)
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
        if device.type=="cuda"
        else False
    )


    print(
        "训练图片:",
        len(train_dataset)
    )

    print(
        "验证图片:",
        len(val_dataset)
    )


    print(
        f"batch={BATCH_SIZE}, epochs={EPOCHS}"
    )



    # =========================
    # 模型
    # =========================


    model = get_model()

    model.to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    training_log=[]



    # =========================
    # Training Loop
    # =========================


    for epoch in range(EPOCHS):


        model.train()


        total_loss=0


        print(
            f"\n开始 Epoch {epoch+1}/{EPOCHS}"
        )


        for step,(images,targets) in enumerate(
            train_loader,
            start=1
        ):


            images=[
                img.to(device)
                for img in images
            ]


            targets=[
                {
                    k:v.to(device)
                    for k,v in t.items()
                }
                for t in targets
            ]


            loss_dict=model(
                images,
                targets
            )


            loss=sum(
                loss
                for loss in loss_dict.values()
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            total_loss += loss.item()



            if step==1 or step%20==0:

                print(
                    f"step:{step}, loss:{loss.item():.6f}"
                )



        train_loss = (
            total_loss /
            max(len(train_loader),1)
        )



        # 保存当前权重

        torch.save(
            model.state_dict(),
            WEIGHTS_PATH
        )



        # =========================
        # 验证
        # =========================


        print("开始验证...")


        metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device
        )



        # 兼容不同版本字段名称

        map50 = metrics.get(
            "mAP50",
            metrics.get(
                "mAP@0.5",
                0
            )
        )


        map5095 = metrics.get(
            "mAP50_95",
            metrics.get(
                "mAP@0.5:0.95",
                0
            )
        )


        precision = metrics.get(
            "precision",
            0
        )


        recall = metrics.get(
            "recall",
            0
        )



        record={

            "epoch":epoch+1,

            "train_loss":train_loss,

            "val_loss":metrics.get(
                "val_loss",
                0
            ),

            "cls_loss":0,

            "box_loss":0,

            "learning_rate":
                optimizer.param_groups[0]["lr"],


            "mAP50":map50,

            "mAP50_95":map5095,

            "precision":precision,

            "recall":recall

        }



        training_log.append(record)

        save_log(training_log)



        print(
            f"""
Epoch {epoch+1}完成
train_loss={train_loss:.6f}
mAP50={map50:.6f}
mAP50_95={map5095:.6f}
"""
        )



    torch.save(
        model.state_dict(),
        WEIGHTS_PATH
    )


    print("\n训练完成")
    print(
        "模型:",
        WEIGHTS_PATH
    )

    print(
        "日志:",
        TRAIN_LOG_PATH
    )



if __name__=="__main__":

    main()
