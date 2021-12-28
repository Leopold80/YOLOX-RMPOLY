# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.data.datasets.rmpoly import RMPOLYDataset
from yolox.exp import PolyExp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # 只有红蓝装甲板被识别到，所以只有两类
        self.num_classes = 2
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1
        self.max_epoch = 300

        # ---------- transform config ------------ #
        # 马赛克增强概率
        self.mosaic_prob = 1.0
        # self.mosaic_prob = 0.6
        # 叠加概率
        self.mixup_prob = 1.0
        # self.mixup_prob = 0.6
        # 颜色变换概率
        self.hsv_prob = 0.0
        # 翻转概率
        self.flip_prob = 0.5
        # 原来是5
        self.multiscale_range = 2
        # self.mosaic_scale = (0.1, 2)
        # self.mixup_scale = (0.5, 1.5)
        # self.shear = 2.0
        self.mosaic_scale = (0.5, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 1.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_num_workers = 0

        self.eval_interval = 10000000000000000000000000000000000000000000000

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            RMCODataset,
            PolyTrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            PolyMosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            # dataset = VOCDetection(
            #     data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            #     image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            #     img_size=self.input_size,
            #     preproc=TrainTransform(
            #         max_labels=50,
            #         flip_prob=self.flip_prob,
            #         hsv_prob=self.hsv_prob),
            #     cache=cache_img,
            # )
            dataset = RMPOLYDataset(
                root_path="H:/DLProject/UCAS_AOD/dataset_utils/dataset",
                # root_path="D:/DLproject/YOLOX-RMPOLY/datasets/UCAS50/images/train",
                img_size=(640, 640),
                preproc=PolyTrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                )
            )

        dataset = PolyMosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=PolyTrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler,
                             "worker_init_fn": worker_init_reset_seed}

        # Make sure each process has different random seed, especially for 'fork' method

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import (
            RMCODataset,
            PolyTrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            PolyMosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            # dataset = VOCDetection(
            #     data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            #     image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            #     img_size=self.input_size,
            #     preproc=TrainTransform(
            #         max_labels=50,
            #         flip_prob=self.flip_prob,
            #         hsv_prob=self.hsv_prob),
            #     cache=cache_img,
            # )
            dataset = RMPOLYDataset(
                root_path="H:/DLProject/UCAS_AOD/data_transform/fucku",
                # root_path="D:/DLproject/YOLOX-RMPOLY/datasets/UCAS50/images/train",
                img_size=(640, 640),
                preproc=PolyTrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                )
            )

        dataset = PolyMosaicDetection(
            dataset,
            # mosaic=not no_aug,
            img_size=self.input_size,
            preproc=PolyTrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            # mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler,
                             "worker_init_fn": worker_init_reset_seed}

        # Make sure each process has different random seed, especially for 'fork' method

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
