from yolox.data.datasets.rmco import RMCODataset

if __name__ == "__main__":
    from pathlib import Path
    from loguru import logger
    from tqdm import tqdm

    dataset = RMCODataset(root_path="G:/RM_CV/rmco/yolo/roco128/roco_train", img_size=(640, 640))
    dataset.pull_item(34)

    logger.debug("")
