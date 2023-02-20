# Author: Yizhak Ben-Shabat (Itzik), 2022
# test 3DInAction on Dfaust dataset

import sys
sys.path.append('../dfaust')

import os
import argparse
import i3d_utils
import sys
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from DfaustDataset import DfaustActionClipsDataset as Dataset
import importlib.util
import visualization
import pathlib
import utils as point_utils
import yaml
from models.pointnet2_cls_ssg import PointNet2, PointNetPP4D, PointNet2Basic
from torch.multiprocessing import set_start_method

from util_scripts.GradCam import GradCam

np.random.seed(0)
torch.manual_seed(0)
# pn2_patchlets_8bs_2steps_skip_connection_2_millbrae, pn2_4d_basic_8bs_2steps_skip_connection_0_millbrae, pn2_patchlets_8bs_2steps_skip_connection_detach_3_millbrae
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='../dfaust/log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='pn2_patchlets_8bs_2steps_skip_connection_detach_3_millbrae', help='unique run identifier')
parser.add_argument('--model_ckpt', type=str, default='000000.pt', help='checkpoint to load')
args = parser.parse_args()

# from pointnet import PointNet4D
def run(cfg, logdir, model_path, output_path):

    dataset_path = cfg['DATA']['dataset_path']
    pc_model = cfg['MODEL']['pc_model']
    batch_size = 1
    frames_per_clip = cfg['DATA']['frames_per_clip']
    n_points = cfg['DATA']['n_points']
    shuffle_points = 'fps_each' #cfg['DATA']['shuffle_points']
    gender = cfg['DATA']['gender']
    subset = cfg['TESTING']['set']
    aug = cfg['TESTING']['aug']
    noisy_data = cfg['DATA']['noisy_data']

    # setup dataset
    test_dataset = Dataset(dataset_path, frames_per_clip=frames_per_clip, set=subset, n_points=n_points, last_op='pad',
                           data_augmentation=aug, shuffle_points=shuffle_points, gender=gender, noisy_data=noisy_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                                  pin_memory=True)
    num_classes = test_dataset.action_dataset.num_classes

    # setup the model
    checkpoints = torch.load(model_path)
    if pc_model == 'pn1':
        spec = importlib.util.spec_from_file_location("PointNet1", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet1"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet1(k=num_classes, feature_transform=True)
    elif pc_model == 'pn1_4d_basic':
        spec = importlib.util.spec_from_file_location("PointNet1Basic", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet1Basic"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet1Basic(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
    elif pc_model == 'pn1_4d':
        spec = importlib.util.spec_from_file_location("PointNet4D", os.path.join(logdir, "pointnet.py"))
        pointnet = importlib.util.module_from_spec(spec)
        sys.modules["PointNet4D"] = pointnet
        spec.loader.exec_module(pointnet)
        model = pointnet.PointNet4D(k=num_classes, feature_transform=True, n_frames=frames_per_clip)
    elif pc_model == 'pn2':
            spec = importlib.util.spec_from_file_location("PointNet2",
                                                          os.path.join(logdir, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_4d':
            spec = importlib.util.spec_from_file_location("PointNetPP4D",
                                                          os.path.join(logdir, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNetPP4D"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNetPP4D(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_4d_basic':
            spec = importlib.util.spec_from_file_location("PointNet2Basic",
                                                          os.path.join(logdir, "pointnet2_cls_ssg.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2Basic"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2Basic(num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == 'pn2_patchlets':
            spec = importlib.util.spec_from_file_location("PointNet2Patchlets",
                                                          os.path.join(logdir, "patchlets.py"))
            pointnet_pp = importlib.util.module_from_spec(spec)
            sys.modules["PointNet2Patchlets"] = pointnet_pp
            spec.loader.exec_module(pointnet_pp)
            model = pointnet_pp.PointNet2Patchlets(cfg=cfg['MODEL']['PATCHLET'], num_class=num_classes, n_frames=frames_per_clip)
    elif pc_model == '3dmfv':
            spec = importlib.util.spec_from_file_location("FourDmFVNet",
                                                          os.path.join(logdir, "pytorch_3dmfv.py"))
            pytorch_3dmfv = importlib.util.module_from_spec(spec)
            sys.modules["FourDmFVNet"] = pytorch_3dmfv
            spec.loader.exec_module(pytorch_3dmfv)
            model = pytorch_3dmfv.FourDmFVNet(n_gaussians=cfg['MODEL']['3DMFV']['n_gaussians'], num_classes=num_classes,
                                              n_frames=frames_per_clip)


    model.load_state_dict(checkpoints["model_state_dict"])  # load trained model
    model.cuda()
    # model = nn.DataParallel(model)
    model.eval()

    # target_layers = [model.patchlet_temporal_conv1]
    target_layers = [model.patchlet_extractor1]
    cam = GradCam.GradCAM(model=model, target_layers=target_layers, reshape_transform=None)
    #
    # Iterate over data.
    for test_batchind, data in enumerate(test_dataloader):

        # get the inputs
        inputs, labels_int, seq_idx, subseq_pad = data['points'], data['labels'], data['seq_idx'], data['padding']
        inputs = inputs.permute(0, 1, 3, 2).cuda().requires_grad_().contiguous()
        labels = F.one_hot(labels_int.to(torch.int64), num_classes).permute(0, 2, 1).float().cuda()

        if test_batchind in [1, 6, 22, 12, 14]:
            cam_result = cam(input_tensor=inputs, targets=labels)
            per_patch_cam = cam_result.mean(-1)
            out_dict = model(inputs)
            patchlet_points = out_dict['patchlet_points']
            visualization.pc_patchlet_points_vis(patchlet_points[0].detach().cpu().numpy(), colors=per_patch_cam[0] )
        # logits = out_dict['pred']
        # pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()

        # compute losses
        # loc_loss = F.binary_cross_entropy_with_logits(logits, labels)
        # cls_loss = F.binary_cross_entropy_with_logits(torch.max(logits, dim=2)[0],
        #                                               torch.max(labels, dim=2)[0])
        # loss = (0.5 * loc_loss + 0.5 * cls_loss)

        # loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').mean(1).squeeze()
        #
        # if test_batchind == 15:
        #     # Compute gradient magnitude
        #     batch_idx = 0
        #     point_grad = utils.gradient(inputs, loss)
        #     point_grad_mag = torch.linalg.norm(point_grad[:, 0:], axis=2)
        #     # point_grad_mag = (point_grad_mag - point_grad_mag.min(axis=2)[0].unsqueeze(-1)) / \
        #     #                  (point_grad_mag.max(axis=2)[0].unsqueeze(-1) - point_grad_mag.min(axis=2)[0].unsqueeze(-1))
        #     point_grad_mag = (point_grad_mag - point_grad_mag.min()) / (point_grad_mag.max() - point_grad_mag.min())
        #     point_grad_mag = torch.clip(point_grad_mag, min=point_grad_mag.mean(axis=2).unsqueeze(-1) - 2 *point_grad_mag.std(axis=2).unsqueeze(-1),
        #                                 max=point_grad_mag.mean(axis=2).unsqueeze(-1) + 2 *point_grad_mag.std(axis=2).unsqueeze(-1))
        #
        #     #Visualize
        #     visualization.pc_seq_vis(inputs[batch_idx, 0:].permute(0, 2, 1).detach().cpu().numpy(), text=None,
        #                              color=point_grad_mag[batch_idx].detach().cpu().numpy(), point_size=15, rgb=False)





if __name__ == '__main__':
    cfg = yaml.safe_load(open(os.path.join(args.logdir, args.identifier, 'config.yaml')))
    logdir = os.path.join(args.logdir, args.identifier)
    output_path = os.path.join(logdir, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(logdir, args.model_ckpt)
    run(cfg, logdir, model_path, output_path)


