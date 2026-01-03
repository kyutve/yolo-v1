import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from config import CFG
from model import YOLK
from data import YOLK_DATASET
from loss import YOLKLoss
from logger import YolkLogger

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--augment", type=str, default=None)

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, val_loader, criterion):
    """Validation loss 계산"""
    model.eval()
    total_loss = 0.0

    for images, targets in val_loader:
        images = images.to(CFG.device)
        targets = targets.to(CFG.device)

        preds = model(images)
        loss = criterion(preds, targets)
        total_loss += loss.item()

    return total_loss / len(val_loader)


def train(resume=False, resume_dir=None):
    print("=== YOLOv1 Training Start ===")

    # ------------------- LOGGER -------------------
    logger = YolkLogger(log_dir="./logs", resume_dir=resume_dir if resume else None)
    logger.save_config(CFG)

    # ------------------- DATASET -------------------
    train_ds = YOLK_DATASET(
        image_dir="./data/images/train",
        label_dir="./data/labels/train",
        config=CFG,
        augment=CFG.augment,
        normalize=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers
    )

    val_ds = YOLK_DATASET(
        image_dir="./data/images/val",
        label_dir="./data/labels/val",
        config=CFG,
        augment=False,
        normalize=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers
    )

    # ------------------- MODEL ---------------------
    model = YOLK().to(CFG.device)
    optimizer = optim.Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    criterion = YOLKLoss()

    start_epoch = 1
    best_val_loss = float("inf")

    # ------------------- RESUME --------------------
    if resume and resume_dir is not None:
        ckpt_path = YolkLogger.find_latest_checkpoint(resume_dir)
        if ckpt_path is not None:
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=CFG.device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            logger.log_txt(f"Resumed training from epoch {start_epoch}")

    # ------------------- TRAIN LOOP ----------------
    for epoch in range(start_epoch, CFG.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.epochs}")

        for step, (images, targets) in enumerate(pbar, start=1):
            images = images.to(CFG.device)
            targets = targets.to(CFG.device)

            preds = model(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            lr = optimizer.param_groups[0]["lr"]

            # CSV 로그 (훈련 step)
            logger.log_csv(epoch, step, loss.item(), lr)

            # tqdm 실시간 업데이트
            pbar.set_postfix({"loss": loss.item(), "lr": lr})

            # 20 step마다 기록
            if step % 20 == 0:
                logger.log_txt(
                    f"[Epoch {epoch}/{CFG.epochs}] Step {step}/{len(train_loader)} | "
                    f"Loss={loss.item():.4f} | LR={lr}"
                )

            # Step checkpoint (200 step마다)
            if step % 200 == 0:
                logger.save_checkpoint(model, optimizer, epoch, step)

        # ---- Epoch Summary ----
        avg_loss = total_loss / len(train_loader)
        logger.log_txt(f"[Epoch {epoch}] Avg Train Loss = {avg_loss:.4f}")

        # ---- Validation ----
        val_loss = evaluate(model, val_loader, criterion)
        logger.log_txt(f"[Epoch {epoch}] Val Loss = {val_loss:.4f}")

        # Best Model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.save_best(model, optimizer, epoch, val_loss)

        # Epoch Checkpoint 저장
        logger.save_checkpoint(model, optimizer, epoch)

    print("=== Training Finished ===")


if __name__ == "__main__":
    args = parse_args()

    # 실행 인자로 넘어온 값 덮어쓰기
    if args.epochs is not None:
        CFG.epochs = args.epochs
    if args.batch is not None:
        CFG.batch_size = args.batch
    if args.lr is not None:
        CFG.lr = args.lr
    if args.device is not None:
        CFG.device = args.device
    if args.augment is not None:
        CFG.augment = (args.augment.lower() == "true")

    train()
    # resume 예시 → train(resume=True, resume_dir="./logs/run_20250101_123000")
