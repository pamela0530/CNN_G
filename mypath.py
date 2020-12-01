class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/user/data/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/home/user/data/datasets/VOC/augdata/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'gastro':
            return '/home/user/pythonwork/fcn.berkeleyvision.org/voc-fcn8s-gastroscope/data'
        elif dataset == 'lung':
            return '/home/user/data/datasets/lungs'
        elif dataset == 'DRIVE':
            return '/home/user/data/datasets/DRIVE'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
