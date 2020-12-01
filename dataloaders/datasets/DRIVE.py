from __future__ import print_function, division

from PIL import Image,ExifTags
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import os
from ..utils import decode_segmap



class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 2
    class_names = np.array([
        'background',
        'vessel',])
    # gas_means = (0.353,0.411,0.676)
    # gas_std = (0.125, 0.169,0.445)
    gas_means = (0.16155026426257252, 0.2681969564344103, 0.5078456500372)

    gas_std = (0.011175970026355841, 0.03434524825000446, 0.12306384809494823)
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('DRIVE'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images')
        self._cat_dir = os.path.join(self._base_dir, 'lab')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:

            img_path_spit = os.path.join(self._base_dir,splt) + "/images"
            lines = os.listdir(img_path_spit)

            for ii, line in enumerate(lines):

                _image = os.path.join(self._image_dir, line[:-3]+"png")
                _cat = os.path.join(self._cat_dir, line[:2]+"_manual1.gif" )
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
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index])

        _target = Image.open(self.categories[index])
        _target = np.asarray(_target)/255.0
        _target=Image.fromarray(_target)
        # _img = cv2.imread(self.images[index])
        # _target = cv2.imread(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.gas_means, std=self.gas_std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            # tr.Normalize(mean=(0.353,0.411,0.676), std=(0.125, 0.169,0.445)),
            tr.Normalize(mean=self.gas_means, std=self.gas_std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def un_trasform_val(self,img,gt):
        img_tmp = np.transpose(img.cpu().numpy(), axes=[1, 2, 0])
        # img_tmp *= (0.125, 0.169, 0.445)
        # img_tmp += (0.353, 0.411, 0.676)
        img_tmp *= self.gas_std
        img_tmp += self.gas_means
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        tmp = np.array(gt).astype(np.uint8)
        segmap = decode_segmap(tmp, dataset='lung')

        # img = img.numpy()
        # # gt = sample['label'].numpy()
        # # tmp = np.array(gt[jj]).astype(np.uint8)
        # # segmap = decode_segmap(tmp, dataset='pascal')
        # # img_tmp = np.transpose(img, axes=[1, 2, 0])
        # img_tmp=img
        # img_tmp *= self.gas_std
        # img_tmp += self.gas_means
        # img_tmp *= 255.0
        # img_tmp = img_tmp.astype(np.uint8)
        return img_tmp,segmap


    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 584
    args.crop_size = 584

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            print(np.max(gt))
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='lung')
            # img_tmp =img[jj]
            # img_tmp=np.transpose(img, axes=[1, 2, 0])
            img_tmp = np.transpose(img[jj])
            # img_tmp *= (0.125, 0.169, 0.445)
            # img_tmp += (0.353, 0.411, 0.676)
            img_tmp *= voc_train.gas_std
            img_tmp += voc_train.gas_means
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            # img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
            # img_tmp = np.transpose(img_tmp)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


