import os
import torch

class YOLK_CFG:
    def __init__(self):
        
        ### ----- 프로젝트 ----- ###
        self.project = "YOLK"
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### ----- 데이터셋 ----- ###
        self.dataset = "bdd100k"
        self.data_root = "./data"
        self.classes = ['car', 'traffic light', 'person', 'truck', 'bus']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}


        ### ----- 학습변수 ----- ###
        self.IMAGE_SIZE = 448
        self.batch_size = 16
        
        self.S = 7
        self.B = 2
        self.C = 5

        self.num_workers = 4
        self.augment = True

        self._normal = {
            "mean": [0.485, 0.456, 0.406],
            "std" : [0.229, 0.224, 0.225]
        }

        ### ----- Model.py ----- ###
        self.backbone = "resnet50"
        self.dropout_prob = 0.3

        ### ----- Loss.py ----- ###
        self.l_coord = 5.0
        self.l_noobj = 0.5

        ### ----- train.py ----- ###
        self.lr = 1e-4
        self.weight_decay= 5e-4
        self.optimizer = "adam"
        self.epochs = 1

        self.lr_decay = True
        self.lr_gamma = 0.95

        ### ----- save.py ----- ###
        self.save_dir = "./checkpoints"
        self.log_dir = "./logs"
        self.result_dir = "./results"


        ### ----- utils.py ----- ###
        self.WARMUP_EPOCHS = 0
        self.EPSILON = 1E-6


        self._prepare_dirs()


    def _prepare_dirs(self):
        """파일 디렉토리가 없을경우 생성"""
        for d in [self.save_dir, self.log_dir, self.result_dir]:
            os.makedirs(d, exist_ok=True)

    def __repr__(self):
        """객체 정보 추출해주는 매서드"""
        rep = "==== YOLK CONFIG ====\n"
        for k, v in self.__dict__.items():
            if not callable(v):
                rep += f"{k:15s}: {v}\n"
        return rep
    
    def normal(self, key):
        return self._normal[key]

CFG = YOLK_CFG()