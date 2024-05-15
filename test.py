# Author: Yizhak Ben-Shabat (Itzik), 2022
# test 3DInAction on Dfaust dataset

import os
import argparse
import i3d_utils
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import build_dataloader
import random
import yaml
import sys
import importlib


import logging

def create_basic_logger(logdir, level = 'info'):
    print(f'Using logging level {level} for train.py')
    global logger
    logger = logging.getLogger('test_logger')
    
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
    file_handler = logging.FileHandler(os.path.join(logdir, "log_test.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    #stream_handler.setLevel(logging.INFO)
    #stream_handler.setFormatter(stream_handler)
    logger.addHandler(stream_handler)
    return logger



def run(cfg, logdir, model_path, output_path, args, logger=None):
    if logger == None:
        logger = create_basic_logger(logdir, 'info')

    batch_size = cfg['TESTING']['batch_size']
    frames_per_clip = cfg['DATA']['frames_per_clip']
    subset = cfg['TESTING']['set']
    pred_output_filename = os.path.join(output_path, subset + '_pred.npy')
    json_output_filename = os.path.join(output_path, subset + '_action_segments.json')
    data_name = cfg['DATA']['name']

    if args.fix_random_seed:
        seed = cfg['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    test_dataloader, test_dataset = build_dataloader(config=cfg, training=False, shuffle=False, logger=logger)
    #test_dataloader, test_dataset = build_dataloader(config=cfg, training=False, shuffle=False)
    num_classes = test_dataset.num_classes

    # setup the model
    spec = importlib.util.spec_from_file_location('build_model_from_logdir', os.path.join(logdir, 'models', '__init__.py'))
    build_model_from_logdir = importlib.util.module_from_spec(spec)
    sys.modules['build_model_from_logdir'] = build_model_from_logdir
    spec.loader.exec_module(build_model_from_logdir)
    model = build_model_from_logdir.build_model_from_logdir(logdir, cfg['MODEL'], num_classes, frames_per_clip).get()

    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    n_examples = 0

    # Iterate over data.
    avg_acc = []
    pred_labels_per_video = [[] for _ in range(test_dataset.get_num_seq())]
    logits_per_video = [[] for _ in range(test_dataset.get_num_seq())]

    for test_batchind, data in enumerate(test_dataloader):
        # get the inputs
        inputs, labels, vid_idx, frame_pad = data['inputs'], data['labels'], data['vid_idx'], data['frame_pad']
        in_channel = cfg['MODEL'].get('in_channel', 3)
        inputs = inputs[:, :, 0:in_channel, :].cuda()
        labels = labels.cuda()

        with torch.no_grad():
            out_dict = model(inputs)

        logits = out_dict['pred']

        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
        avg_acc.append(acc.detach().cpu().numpy())
        n_examples += batch_size
        logger.info('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()
        logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()

        pred_labels_per_video, logits_per_video = \
            utils.accume_per_video_predictions(vid_idx, frame_pad, pred_labels_per_video, logits_per_video,
                                               pred_labels, logits, frames_per_clip)

    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]

    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list, dataset_name=data_name)

def main(args):
    cfg = yaml.safe_load(open(os.path.join(args.logdir, args.identifier, 'config.yaml')))
    logdir = os.path.join(args.logdir, args.identifier)
    output_path = os.path.join(logdir, 'results')
    os.makedirs(output_path, exist_ok=True)
    
    logger = create_basic_logger(logdir = logdir, level = args.loglevel)
    
    model_path = os.path.join(logdir, args.model_ckpt)
    
    logger.info(f'=================== Starting testing run for {args.identifier}')
    logger.info(f'Config: {cfg}')
    logger.info(f'Model Path: {model_path}')
    run(cfg, logdir, model_path, output_path, args, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
    parser.add_argument('--loglevel', type=str, default='info', help='set level of logger')
    parser.add_argument('--identifier', type=str, default='debug', help='unique run identifier')
    parser.add_argument('--model_ckpt', type=str, default='000000.pt', help='checkpoint to load')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
    args = parser.parse_args()
    
    main(args)
