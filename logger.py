import os
import csv
import json
import datetime
import torch
from glob import glob
import re


class YolkLogger:
    def __init__(self, log_dir="./logs", resume_dir=None):
        """
        resume_dir이 주어지면 기존 세션 폴더 재사용
        """
        if resume_dir is None:
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(log_dir, f"run_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)
        else:
            self.session_dir = resume_dir

        # 파일 경로 지정
        self.txt_path = os.path.join(self.session_dir, "train_log.txt")
        self.csv_path = os.path.join(self.session_dir, "metrics.csv")
        self.config_path = os.path.join(self.session_dir, "config.json")

        # CSV 헤더 생성 (resume 시 스킵)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "step", "loss", "lr"])

    # ---------------------- TEXT LOG ----------------------
    def log_txt(self, msg: str):
        print(msg)
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # ---------------------- CSV LOG ------------------------
    def log_csv(self, epoch, step, loss, lr):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, lr])

    # ---------------------- CONFIG SAVE --------------------
    def save_config(self, cfg):
        cfg_dict = {}

        for k, v in cfg.__dict__.items():
            if callable(v):
                continue

            # torch.device → 문자열 변환
            if isinstance(v, torch.device):
                cfg_dict[k] = str(v)
            else:
                cfg_dict[k] = v

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=4)

        self.log_txt("Saved config.json")

    # ---------------------- CHECKPOINT ---------------------
    def save_checkpoint(self, model, optimizer, epoch, step=None):
        """epoch별 or step별 체크포인트 저장"""
        if step is None:
            name = f"checkpoint_epoch_{epoch}.pth"
        else:
            name = f"checkpoint_epoch_{epoch}_step_{step}.pth"

        path = os.path.join(self.session_dir, name)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
            },
            path
        )
        self.log_txt(f"Checkpoint saved: {path}")

    # ---------------------- BEST MODEL ---------------------
    def save_best(self, model, optimizer, epoch, val_loss):
        path = os.path.join(self.session_dir, "best_model.pth")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            path
        )
        self.log_txt(f"[BEST] Epoch {epoch} updated | val_loss={val_loss:.4f}")

    # ---------------------- RESUME 기능 ---------------------
    @staticmethod
    def extract_epoch(filename):
        """checkpoint_epoch_X.pth → X 숫자만 추출"""
        match = re.search(r"checkpoint_epoch_(\d+)", filename)
        if match:
            return int(match.group(1))
        return -1

    @staticmethod
    def find_latest_checkpoint(session_dir):
        ckpts = glob(os.path.join(session_dir, "checkpoint_epoch_*.pth"))
        if len(ckpts) == 0:
            return None

        # epoch 번호 기준 정렬
        ckpts = sorted(ckpts, key=lambda x: YolkLogger.extract_epoch(x))
        return ckpts[-1]
