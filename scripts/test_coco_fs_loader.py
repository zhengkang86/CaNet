import os
import tqdm
from torch.utils.data import DataLoader

from fss import FewShotInstData
import sys
sys.path.append(".")
from coco_for_canet import COCO_CaNet


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


if __name__ == "__main__":
    coco_dir = '/home/kang/Projects/data/COCO'
    exclude_files = './json/bad_inst_list_train.json'
    folds = [0, 1, 2, 3]

    dataset = FewShotInstData(coco_dir, 'train2017', folds, 1, exclude_list_file=exclude_files, img_size=112)

    for i in tqdm.tqdm(range(len(dataset))):
        item = dataset[i]

    # train_loader = DataLoader(dataset, batch_size=24, num_workers=24)
    # for i, batch_data in enumerate(tqdm.tqdm(train_loader)):
    #     query_inst, support_insts = batch_data

    # dataset = COCO_CaNet(coco_dir, 'train2017', folds, 1, exclude_list_file=exclude_files, normalize_mean=IMG_MEAN, normalize_std=IMG_STD)
    # train_loader = DataLoader(dataset, batch_size=24, num_workers=24)
    # for i, batch_data in enumerate(tqdm.tqdm(train_loader)):
    #     query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index = batch_data
