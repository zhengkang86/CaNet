import os
import sys
import tqdm
import matplotlib.pyplot as plt

from fss import FewShotInstData
sys.path.append('.')
from coco_for_canet import COCO_CaNet


if __name__ == '__main__':
    coco_dir = '/home/kang/Projects/data/COCO'

    # FSDataset = FewShotInstData(coco_dir, 'val2017', [0], 1, img_size=224)
    # # FSDataset = FewShotInstData(coco_dir, 'train2017', [0], 1, img_size=224)
    # print(len(FSDataset))
    # for i in tqdm.tqdm(range(len(FSDataset))):
    #     query_inst, support_insts = FSDataset[i]
    #     try:
    #         assert(query_inst['roi'].shape[-1] == 3)
    #     except:
    #         print(query_inst['roi'].shape)

    # # fig, ax = plt.subplots(1, 2)
    # # ax[0].imshow(query_inst['roi'])
    # # ax[1].imshow(query_inst['roi_mask'], cmap='gray')
    # # plt.show()
    # import pdb; pdb.set_trace()

    FSDataset = COCO_CaNet(coco_dir, 'train2017', [0], 1)
    import pdb; pdb.set_trace()
    for i in range(len(FSDataset)):
        query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index = FSDataset[i]
        import pdb; pdb.set_trace()
        assert(query_rgb.shape[0] == 1)
