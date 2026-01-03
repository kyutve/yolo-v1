import os
import random
import json
from PIL import Image

import torch
import utils
from config import *
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class JSONDetection(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # 이미지 파일 리스트
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

        # (img_path, json_path) 페어링
        self.samples = [
            (
                os.path.join(image_dir, f),
                os.path.join(label_dir, f.replace(".jpg", ".json"))
            )
            for f in self.image_files
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load JSON label
        with open(json_path, "r") as f:
            label = json.load(f)

        # Transform
        if self.transform:
            image = self.transform(image)

        return image, label
    
class YOLK_DATASET(Dataset):
    def __init__(self,image_dir, label_dir, config, augment=False, normalize=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment
        self.normalize = normalize
        self.config = config
        self.classes = CFG.classes
        self.class_to_idx = CFG.class_to_idx

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

        self.dataset = JSONDetection(
            image_dir=image_dir,
            label_dir=label_dir,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            ])
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, raw_label = self.dataset[idx]

        # ---------------------------------------------
        # BDD100K JSON 
        # ---------------------------------------------
        objects = raw_label["frames"][0]["objects"]

        parsed_boxes = []
        for obj in objects:
            name = obj["category"]
            if name not in self.classes:
                continue

            if "box2d" not in obj:
                continue

            box = obj["box2d"]
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            parsed_boxes.append((name, (x1, x2, y1, y2)))

        # ---------------------------------------------
        # Augmentation
        # ---------------------------------------------
        if self.augment:
            x_shift = int((0.2 * random.random() - 0.1) * CFG.IMAGE_SIZE)
            y_shift = int((0.2 * random.random() - 0.1) * CFG.IMAGE_SIZE)
            scale = 1 + 0.2 * random.random()

            image = TF.affine(image, angle=0, scale=scale,
                              translate=(x_shift, y_shift), shear=0)

            image = TF.adjust_hue(image, 0.2 * random.random() - 0.1)
            image = TF.adjust_saturation(image, 0.2 * random.random() + 0.9)
        else:
            x_shift = y_shift = 0
            scale = 1.0

        if self.normalize:
            image = TF.normalize(image,
                                 mean=CFG.normal("mean"),
                                 std=CFG.normal("std"))

        # ---------------------------------------------
        # GRID 설정
        # ---------------------------------------------
        grid_size_x = CFG.IMAGE_SIZE / CFG.S
        grid_size_y = CFG.IMAGE_SIZE / CFG.S

        depth = 5 * CFG.B + CFG.C
        target = torch.zeros((CFG.S, CFG.S, depth))

        boxes_used = {}
        class_used = {}

        # ---------------------------------------------
        # Target 만들기
        # ---------------------------------------------
        for name, (x1, x2, y1, y2) in parsed_boxes:
            class_idx = self.class_to_idx[name]

            # Augment bbox
            if self.augment:
                half_w = CFG.IMAGE_SIZE / 2
                half_h = CFG.IMAGE_SIZE / 2

                x1 = utils.scale_bbox_coord(x1, half_w, scale) + x_shift
                x2 = utils.scale_bbox_coord(x2, half_w, scale) + x_shift
                y1 = utils.scale_bbox_coord(y1, half_h, scale) + y_shift
                y2 = utils.scale_bbox_coord(y2, half_h, scale) + y_shift

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            col = int(cx // grid_size_x)
            row = int(cy // grid_size_y)

            if not (0 <= col < CFG.S and 0 <= row < CFG.S):
                continue

            grid_cell = (row, col)

            # class one-hot
            one_hot = torch.zeros(CFG.C)
            one_hot[class_idx] = 1.0
            target[row, col, :CFG.C] = one_hot

            if grid_cell not in boxes_used:
                boxes_used[grid_cell] = 0

            b_idx = boxes_used[grid_cell]
            if b_idx >= CFG.B:
                continue

            w = (x2 - x1) / CFG.IMAGE_SIZE
            h = (y2 - y1) / CFG.IMAGE_SIZE

            tx = (cx - col * grid_size_x) / grid_size_x
            ty = (cy - row * grid_size_y) / grid_size_y

            bbox = torch.tensor([tx, ty, w, h, 1.0])

            start = CFG.C + 5 * b_idx
            target[row, col, start:start+5] = bbox

            boxes_used[grid_cell] += 1

        return image, target

class YOLK_DATASET2(Dataset):
    def __init__(self,image_dir, label_dir, config, augment=False, normalize=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment
        self.normalize = normalize
        self.config = config
        self.classes = CFG.classes
        self.class_to_idx = CFG.class_to_idx

        # 이미지 파일명 리스트
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        
        self.dataset = JSONDetection(
            image_dir=image_dir,
            label_dir=label_dir,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
            ])
        )

    def __len__(self):
        '''이미지 파일 개수'''
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, raw_label = self.dataset[idx]

        # JSON → VOC-style dict 변환
        label = utils.load_label_json(raw_label)


        ### ----- Augmentation ----- ###
        x_shift = int((0.2 * random.random() - 0.1) * CFG.IMAGE_SIZE)
        y_shift = int((0.2 * random.random() - 0.1) * CFG.IMAGE_SIZE)

        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            # 이미지 이동 : 이미지 사이즈 비율 약 -0.1 ~ 0.1 ( 나머지 영역 0 )
            image = TF.affine(image, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            
            # 색조 변환 : 약 -0.1 ~ 0.1
            image = TF.adjust_hue(image, 0.2 * random.random() - 0.1)
            
            # 채도 변환 : 약 0.9 ~ 1.1
            image = TF.adjust_saturation(image, 0.2 * random.random() + 0.9)

        if self.normalize:
            image = TF.normalize(image, mean = CFG.normal("mean"), std = CFG.normal("std"))


        ### ----- Grid Cell ----- ###
        grid_size_x = CFG.IMAGE_SIZE / CFG.S
        grid_size_y = CFG.IMAGE_SIZE / CFG.S

        ### ----- Object Label ----- ###
        boxes = {}
        class_names = {}
        depth = 5 * CFG.B + CFG.C
        target = torch.zeros((CFG.S, CFG.S, depth))

        # get_bounding_boxes : label 데이터를 받아서 (name, coords) 객체 리스트 만듬 ( 없는 객체는 스킵 기능 포함 )
        for j, bbox_pair in enumerate(utils.get_bounding_boxes_json(label)):
            name, coords = bbox_pair

            assert name in self.classes, f"Unrecognized class '{name}'"

            class_idx = self.class_to_idx[name]
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = CFG.IMAGE_SIZE / 2
                half_height = CFG.IMAGE_SIZE / 2

                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            cx = (x_max + x_min) / 2
            cy = (y_max + y_min) / 2
            col = int( cx // grid_size_x ) # 몇 번째 가로 그리드에 있니
            row = int( cy // grid_size_y ) # 몇 번째 세로 그리드에 있니

            if 0 <= col < CFG.S and 0 <= row < CFG.S:
                grid_cell = (row, col)

                # grid cell이 중복이 아니거나 같은 객체가 이미 있을 때 
                if grid_cell not in class_names or name == class_names[grid_cell]:

                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(CFG.C)
                    one_hot[class_idx] = 1.0
                    target[row, col, :CFG.C] = one_hot
                    class_names[grid_cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(grid_cell, 0)
                    if bbox_index < CFG.B:
                        bbox_truth = (
                            (cx - col * grid_size_x) / grid_size_x,     # X coord relative to grid square
                            (cy - row * grid_size_y) / grid_size_y,     # Y coord relative to grid square
                            (x_max - x_min) / CFG.IMAGE_SIZE,                 # Width
                            (y_max - y_min) / CFG.IMAGE_SIZE,                 # Height
                            1.0                                                     # Confidence
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 5 * bbox_index + CFG.C
                        target[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(CFG.B - bbox_index)
                        boxes[grid_cell] = bbox_index + 1
        # image : Tensor(3,448,448), target : (S, S, 5B + C)
        return image, target