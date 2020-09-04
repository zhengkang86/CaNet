import random
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import skimage.transform
import numpy as np
from collections.abc import Iterable

from fss import FewShotInstData


class Resize(object):
    def __init__(self, size, order):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size
        self.order = order

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size, order=self.order)
        # the resize will return a float64 array
        return skimage.util.img_as_ubyte(resize_image)


class COCO_CaNet(FewShotInstData):
    def __init__(self, coco_dir, subset, folds, n_shots, img_size=321, exclude_list_file=None, prob=0.7, normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1]):
        super(COCO_CaNet, self).__init__(coco_dir, subset, folds, n_shots, img_size=img_size, exclude_list_file=exclude_list_file, resize_flag=False)
        all_cat_ids = self.coco.getCatIds()

        active_cat_ids = list()
        for fold in folds:
            active_cat_ids += all_cat_ids[fold*20:(fold+1)*20]
        self.cat_id_mapping = dict(zip(active_cat_ids, list(range(len(active_cat_ids)))))
        self.history_mask_list = [None] * self.__len__()
        self.prob = prob
        self.input_size = [img_size, img_size]
        self.initialize_transformation(normalize_mean, normalize_std, self.input_size)

    def initialize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = transforms.ToTensor()
        # self.resize = transforms.Resize(input_size)
        self.normalize = transforms.Normalize(normalize_mean, normalize_std)

    def flip(self, flag, img):
        if flag > 0.5:
            # print(img.shape)
            if len(img.shape) == 3:
                return img[:, ::-1, :]
            else:
                return img[:, ::-1]
        else:
            return img

    def __getitem__(self, idx):
        query_inst_id = self.inst_ids[idx]
        query_inst = self.getInstByID(query_inst_id)

        same_cat_inst_ids = self.cat_inst_ids[query_inst["inst_cat_id"]].copy()
        same_cat_inst_ids.remove(query_inst_id)
        support_inst_ids = random.choices(same_cat_inst_ids, k=self.n_shots)
        support_insts = [self.getInstByID(inst_id) for inst_id in support_inst_ids]

        if self.history_mask_list[idx] is None:
            history_mask = torch.zeros(2, 41, 41).fill_(0.0)
        else:
            if random.random() > self.prob:
                history_mask = self.history_mask_list[idx]
            else:
                history_mask = torch.zeros(2, 41, 41).fill_(0.0)
        sample_class = self.cat_id_mapping[query_inst['inst_cat_id']]

        query_rgb, query_mask = query_inst['roi'], query_inst['roi_mask']
        support_rgb, support_mask = support_insts[0]['roi'], support_insts[0]['roi_mask']

        # random scale and crop for support
        input_size = self.input_size[0]
        scaled_size = int(random.uniform(1, 1.5) * input_size)

        # scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        # scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_mask = Resize(scaled_size, 0)
        scale_transform_rgb = Resize(scaled_size, 1)
        flip_flag = random.random()
        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag, support_rgb))))
        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag, support_mask)))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        # random scale and crop for query
        scaled_size = input_size  # random.randint(323, 350)
        # scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        # scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_mask = Resize(scaled_size, 0)
        scale_transform_rgb = Resize(scaled_size, 1)
        flip_flag = 0  # random.random()

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag, query_rgb))))
        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag, query_mask)))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        return query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, idx
        # return query_inst, support_insts
        # return query_inst['roi'], query_inst['roi_mask'], support_insts[0]['roi'], support_insts[0]['roi_mask'], \
        #     history_mask, sample_class, idx
