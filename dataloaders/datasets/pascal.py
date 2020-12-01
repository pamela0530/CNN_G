from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21
    voc_means = (0.485, 0.456, 0.406)
    voc_std=(0.229, 0.224, 0.225)
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])


    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target,_name = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample),_name
            elif split == 'val':
                return self.transform_val(sample),_name


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index]).convert("RGB")
        _name = self.images[index]
        def image2label(im):
            """
            vis_img to np.array, transform label formation
            :param im:
            :return:
            """
            colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                        [0, 192, 0], [128, 192, 0], [0, 64, 128]]

            cm2lbl = np.zeros(256 ** 3)
            for i, cm in enumerate(colormap):
                cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
            data = np.array(im, dtype='int32')
            idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
            return np.array(cm2lbl[idx], dtype='int64')
        _target = image2label(_target)
        _target = Image.fromarray(np.uint8(_target))



        return _img, _target,_name

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        # img,label = composed_transforms(sample)

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        # img, label = composed_transforms(sample)
        #
        # def image2label(im):
        #     """
        #     vis_img to np.array, transform label formation
        #     :param im:
        #     :return:
        #     """
        #     colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        #                 [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        #                 [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        #                 [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
        #                 [0, 192, 0], [128, 192, 0], [0, 64, 128]]
        #
        #     cm2lbl = np.zeros(256 ** 3)
        #     for i, cm in enumerate(colormap):
        #         cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        #     data = np.array(im, dtype='int32')
        #     idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        #     return np.array(cm2lbl[idx], dtype='int64')

        # label = image2label(label)
        return composed_transforms(sample)



    def un_trasform_val(self,img):
        img = img.cpu().numpy()
        # gt = sample['label'].numpy()
        # tmp = np.array(gt[jj]).astype(np.uint8)
        # segmap = decode_segmap(tmp, dataset='pascal')
        img_tmp = np.transpose(img, axes=[1, 2, 0])
        # img_tmp *= (0.125, 0.169, 0.445)
        # img_tmp += (0.353, 0.411, 0.676)
        img_tmp *= self.voc_std
        img_tmp += self.voc_means
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        return img_tmp

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


