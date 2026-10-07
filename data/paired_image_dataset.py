from torch.utils import data as data
from torchvision.transforms.functional import normalize
import copy

from data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file,paths_from_folder
from data.transforms import augment, paired_random_crop
from utils import FileClient, imfrombytes, img2tensor
from utils.matlab_functions import rgb2ycbcr
from utils.registry import DATASET_REGISTRY

import numpy as np

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """读取预先准备的低质量图像与目标图像配对。

    支持磁盘目录、LMDB 和配对元信息文件。训练时执行成对随机裁剪、
    翻转及转置增强，再将 BGR/HWC 转换为 RGB/CHW 张量。
    当前 __getitem__ 不执行位深量化或位平面分解。
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt

        self.file_client = None
        # 这两个位深属性未参与当前图像读取与预处理流程。
        self.HBD_depth=16
        self.LBD_depth=1
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)

        elif not self.lq_folder:
            print('******paired_image_dataset.py line71')
            self.paths = paths_from_folder(self.gt_folder)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl, self.task)

    def __getitem__(self, index):
        if self.file_client is None:

            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        img_bytes_1 = self.file_client.get(gt_path, 'gt')
        img_bytes_2 = self.file_client.get(lq_path, 'lq')

        # 按 imfrombytes 的实际解码和固定 65535 缩放逻辑读取目标图像。
        img_gt = imfrombytes(img_bytes_1, float32=True)
        img_lq = imfrombytes(img_bytes_2, float32=True)

        # 对目标和输入应用相同的随机裁剪及数据增强。
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            img_gt, img_lq = paired_random_crop(img_gt, img_lq,gt_size, scale, gt_path)

            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # 当前 y 配置分支只打印提示，没有执行颜色空间转换。
        if 'color' in self.opt and self.opt['color'] == 'y':
            print('*******error in paired_image_dataset')

        # 验证和测试时，按输入尺寸与 scale 裁剪目标图像。
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # 从 BGR/HWC 转换为 RGB/CHW 浮点张量。
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt,'lq_path': gt_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
