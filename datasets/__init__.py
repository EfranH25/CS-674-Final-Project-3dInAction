import torch
from torch.utils.data import DataLoader

from .IKEAEgoDatasetClips import IKEAEgoDatasetClips
from .DfaustDataset import DfaustActionClipsDataset
from .IKEAActionDatasetClips import IKEAActionDatasetClips
from .MSRAction3DDataset import MSRAction3DDataset

import i3d_utils as utils
import logging

def create_basic_logger(logdir, level = 'info'):
    print(f'Using logging level {level} for train.py')
    global logger
    logger = logging.getLogger('train_logger')
    
    #? set logging level
    if level.lower() == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level.lower() == 'info':
        logger.setLevel(logging.INFO)
    elif level.lower() == 'warning':
        logger.setLevel(logging.WARNING)
    elif level.lower() == 'error':
        logger.setLevel(logging.ERROR)
    elif level.lower() == 'critical':
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)
    
    #? create handlers
    file_handler = logging.FileHandler(os.path.join(logdir, "log_train.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    #stream_handler.setLevel(logging.INFO)
    #stream_handler.setFormatter(stream_handler)
    logger.addHandler(stream_handler)
    return logger

def build_dataset(cfg, training=True):
    split = 'train' if training else 'test'
    cfg_data = cfg['DATA']
    if cfg_data.get('name') == 'DFAUST':
        data_augmentation = cfg['TRAINING'].get('aug') if split == 'train' else cfg['TESTING'].get('aug')
        dataset = DfaustActionClipsDataset(
            action_dataset_path=cfg_data['dataset_path'], frames_per_clip=cfg_data['frames_per_clip'], set=split,
            n_points=cfg_data['n_points'], shuffle_points=cfg_data['shuffle_points'], gender=cfg_data['gender'],
            data_augmentation=data_augmentation, noisy_data=cfg_data['noisy_data'],
        )
    elif cfg_data.get('name') == 'IKEA_ASM':
        dataset = IKEAActionDatasetClips(dataset_path=cfg_data['dataset_path'], set=split)
    elif cfg_data.get('name') == 'IKEA_EGO':
        dataset = IKEAEgoDatasetClips(dataset_path=cfg_data['dataset_path'], set=split, cfg_data=cfg_data)
    elif cfg_data.get('name') == 'MSR-Action3D':
        dataset = MSRAction3DDataset(dataset_path=cfg_data['dataset_path'], set=split, cfg_data=cfg_data)
    else:
        raise NotImplementedError
    return dataset


def build_dataloader(config, training=True, shuffle=False, logger=None):
    dataset = build_dataset(config, training)

    num_workers = config['num_workers']
    batch_size = config['TRAINING'].get('batch_size') if training else config['TESTING'].get('batch_size')
    data_sampler = config['DATA'].get('data_sampler')

    split = 'train' if training else 'test'

    if logger == None:
        logger = create_basic_logger(logdir = config, level = 'info')
        logger.info(f"Number of clips in the {split} set: {len(dataset)}")
    else:
        logger.info("Number of clips in the {} set: {}".format(split, len(dataset)))

    if training and data_sampler == 'weighted':
        if config['DATA'].get('name') == 'DFAUST':
            weights = dataset.make_weights_for_balanced_classes()
        elif config['DATA'].get('name') == 'IKEA_ASM' or config['DATA'].get('name') == 'IKEA_EGO':
            weights = utils.make_weights_for_balanced_classes(dataset.clip_set, dataset.clip_label_count)
        elif config['DATA'].get('name') == 'MSR-Action3D':
            weights = dataset.make_weights_for_balanced_classes()
        else:
            raise NotImplementedError
        logger.info(f"Using {config['DATA'].get('name')} for training")
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=lambda x: x,
        batch_size=batch_size,
        #pin_memory=True, #pins to CUDA
    )
    return dataloader, dataset
