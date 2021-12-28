import cv2
import torch
from loguru import logger
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from pathlib import Path
from .datasets_wrapper import Dataset

import random
import numpy as np


class RMCODataset(Dataset):
    # @logger.catch()
    def __init__(self, root_path, img_size, preproc):
        super().__init__(img_size)
        self.preproc = preproc
        self.img_size = img_size
        # 数据集根目录
        self.path = Path(root_path)
        assert self.path.exists(), "训练集他妈的文件夹路径弄对了马？"

        # 训练集所有图片文件的路径
        train_img_dirs = self.path.glob("*.jpg")
        # 训练集所有标注文件的路径
        train_anno_dirs = self.path.glob("*.txt")

        train_dirs = list(zip(train_img_dirs, train_anno_dirs))

        # 打乱数据集
        random.shuffle(train_dirs)

        # 转化标注信息的lambda方法
        read_anno = (
            lambda path: tuple({"cls": c, "x": x, "y": y, "w": w, "h": h} for c, x, y, w, h in
                               (obj.split(" ") for obj in path.open().read().splitlines()))
        )

        bar = tqdm(train_dirs)
        # 生成训练集图片路径+标注信息
        self.annos = [(d, read_anno(a)) for d, a in bar]
        bar.close()

    def __len__(self):
        return len(self.annos)

    def pull_item(self, index):
        obj = self.annos[index]
        img_path = obj[0]
        gts = obj[1]
        gts = [[float(gt['x']), float(gt['y']), float(gt['w']), float(gt['h']), float(gt['cls'])] for gt in gts]
        gts = np.array(gts)

        img = cv2.imread(str(img_path))
        assert img is not None, "读取img错误"
        img_shape = img.shape[:2]

        # 将归一化后的数据取消归一化
        gts[:, 0] *= img_shape[1]
        gts[:, 2] *= img_shape[1]
        gts[:, 1] *= img_shape[0]
        gts[:, 3] *= img_shape[0]

        gts_xywh = gts.copy()
        # 转换为xyxy格式
        gts[:, 0] = gts_xywh[:, 0] - 0.5 * gts_xywh[:, 2]
        gts[:, 1] = gts_xywh[:, 1] - 0.5 * gts_xywh[:, 3]
        gts[:, 2] = gts_xywh[:, 0] + 0.5 * gts_xywh[:, 2]
        gts[:, 3] = gts_xywh[:, 1] + 0.5 * gts_xywh[:, 3]

        img, r = self._reshape_img(img)
        gts[:, :4] *= r

        # boxs = gts[:, :4]
        # for b in boxs:
        #     xmin, ymin, xmax, ymax = b.astype(np.int)
        #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 255), thickness=3)
        # cv2.imshow("vis", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        index = torch.tensor(index) if isinstance(index, int) else index

        return img, gts.copy(), img_shape, index

    def load_anno(self, index):
        obj = self.annos[index]
        gts = obj[1]
        gts = [[float(gt['x']), float(gt['y']), float(gt['w']), float(gt['h']), float(gt['cls'])] for gt in gts]
        gts = np.array(gts)
        return gts

    def _reshape_img(self, img):
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        return resized_img, r

    def __getitem__(self, index) -> T_co:
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, *args, **kwargs):
        return 0, 0

# if __name__ == "__main__":
#     dataset = RMCODataset("G:/RM_CV/rmco/yolo/roco")
