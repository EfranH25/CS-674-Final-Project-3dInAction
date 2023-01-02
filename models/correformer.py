import os.path
import sys
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import cosine_similarity
from models.point_transformer_pytorch import PointTransformerLayer
from models.point_transformer_repro import PointTransformerSeg
from models.pointnet_sem_seg import PNSeg


class CorreFormer(nn.Module):
    def __init__(self, d_model=3, nhead=4, num_encoder_layers=4, num_decoder_layers=6, dim_feedforward=256,
                 twosided=True, transformer_type='none', n_points=1024, loss_type='l2', cat_points=True):
        super(CorreFormer, self).__init__()
        self.pointencoder = PointEncoder(d_model, cat_points)

        self.transformer_type = transformer_type
        self.loss_type = loss_type

        if 'ce' in loss_type:
            self.criterion = torch.nn.CrossEntropyLoss()
        if self.transformer_type == 'plr':
            # only the 16 nearest neighbors would be attended to for each point
            self.transformer = PointTransformerLayer(dim=dim_feedforward,  pos_mlp_hidden_dim=d_model,
                                                      attn_mlp_hidden_mult=nhead, num_neighbors=64)
        elif self.transformer_type == 'ptr':
            self.transformer = PointTransformerSeg(nneighbor=32, npoints=n_points, nblocks=4,
                                                   n_c=dim_feedforward, d_points=3, transformer_dim=dim_feedforward)
        elif self.transformer_type == 'pnseg':
            self.transformer = PNSeg(n_points)
        else:
            self.transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                                    batch_first=True, dropout=0.0)
        self.twosided = twosided

    def single_pass(self, x):
        points = x.permute(0, 2, 1)
        if self.transformer_type == 'plr':
            x, point_features = self.pointencoder(points)
            out = self.transformer(x.permute(0, 2, 1), points.permute(0, 2, 1))
        elif self.transformer_type == 'ptr':
            out = self.transformer(points.permute(0, 2, 1))
        elif self.transformer_type == 'pnseg':
            out, _ = self.transformer(points)
        else:
            x, point_features = self.pointencoder(points)
            out = self.transformer(x, x)
        return out, point_features

    def compute_sim_mat_full(self, out1, out2):
        sim_mat = cosine_similarity(out2, out1)
        corr21 = F.softmax(sim_mat, dim=-1)
        # _, max_ind = torch.max(corr, dim=-1)
        with torch.no_grad():
            max_ind21 = torch.argmax(corr21, dim=-1)
            if self.twosided:
                corr12 = F.softmax(sim_mat, dim=-2)
                max_ind12 = torch.argmax(corr12, dim=-2)
            else:
                max_ind12 = []
        return sim_mat, corr21, max_ind12, max_ind21

    def compute_sim_partial(self, out1, out2, point_ids):
        out1_normalized = out1 / torch.norm(out1, dim=-1, keepdim=True)
        out_2_normalized = out2 / torch.norm(out2, dim=-1, keepdim=True)

        sim_mat = (out_2_normalized[:, point_ids, :]*out1_normalized).sum(-1)

        bad_ids = torch.roll(point_ids, np.random.randint(1, len(point_ids)-1))
        bad_mat = (out_2_normalized[:, bad_ids, :] * out1_normalized).sum(-1)
        sim_mat = torch.cat([sim_mat, bad_mat])
        return sim_mat

    def compute_corr_loss(self, gt_corr, corr, point_ids, out1, out2, sim_mat, point_features1, point_features2):
        # compute correspondance loss
        b, n1, n2 = gt_corr.shape
        loss_dict = {}
        regularizer = (1 - point_features1.square().mean(1)).square().mean() + (1 - point_features2.square().mean(1)).square().mean()
        loss_dict['losses/feature_reg'] = regularizer.cpu().detach().numpy()
        if self.loss_type == 'l2':
            l2_loss = (gt_corr - corr).square()
            l2_mask = torch.max(gt_corr, 1.0 * (torch.rand(b, n1, n2).cuda() < gt_corr.mean())).bool()
            # l2_loss = (l2_mask * l2_loss)
            l2_loss = l2_loss[l2_mask]
            loss_dict['losses/l2_loss'] = l2_loss.cpu().detach().numpy()
            loss = l2_loss.mean() + regularizer

        elif self.loss_type == 'ce':
            # l2_features = (out1[:, point_ids] - out2).square().mean()
            ce_loss = self.criterion(corr.reshape(-1, corr.shape[-1]), point_ids.repeat(b))
            loss_dict['losses/ce_loss'] = ce_loss.cpu().detach().numpy()
            loss = ce_loss + regularizer #+ l2_features
        # elif self.loss_type == 'ce2':
        #     TODO fix ce2
        #     ce_loss1 = self.criterion(corr.reshape(-1, corr.shape[-1]), point_ids.repeat(b))
        #     ce_loss2 = self.criterion(F.softmax(sim_mat, dim=-2).reshape(-1, corr.shape[-1]),
        #                                torch.arange(n1, device=sim_mat.device).repeat(b))
        #     loss_dict['losses/ce_loss'] = ce_loss1.cpu().detach().numpy()
        #     loss_dict['losses/ce2_loss'] = ce_loss2.cpu().detach().numpy()
        #     loss = ce_loss1 + ce_loss2 + regularizer
        elif self.loss_type == 'ce_bbl':
            ce_loss1 = self.criterion(corr.reshape(-1, corr.shape[-1]), point_ids.repeat(b))
            # ce_loss2 = self.criterion(F.softmax(sim_mat, dim=-2).reshape(-1, corr.shape[-1]),
            #                            torch.arange(n1, device=sim_mat.device).repeat(b))
            bbl_loss = self.BBL_loss(sim_mat, thresh=0.8)
            loss_dict['losses/bbl_loss'] = bbl_loss.cpu().detach().numpy()
            loss = ce_loss1  + bbl_loss + regularizer #+ ce_loss2
        loss_dict['losses/total_loss'] = loss.cpu().detach().numpy()
        return loss, loss_dict

    def BBL_loss(self, sim_mat, thresh=0.1):
        # compute best buddy loss
        buddy_pair_mat = sim_mat > thresh
        BBM = (buddy_pair_mat*buddy_pair_mat.permute(0, 2, 1)).float()  # best buddy matrix
        #sym_loss = (BBM - BBM.permute(0, 2, 1)).square().mean()
        ort_loss = (torch.bmm(BBM, BBM.permute(0, 2, 1)) - torch.eye(BBM.shape[-1], device=BBM.device)).square().mean()
        bbl_loss =  ort_loss #+ sym_loss
        return bbl_loss

    def forward(self, x1, x2):
        out1, point_features1 = self.single_pass(x1)
        out2, point_features2 = self.single_pass(x2)

        sim_mat, corr21, max_ind12, max_ind21 = self.compute_sim_mat_full(out1, out2)

        return {'out1': out1, 'out2': out2, 'corr_mat': corr21, 'corr_idx12': max_ind12, 'corr_idx21': max_ind21,
                'sim_mat': sim_mat, 'point_features1': point_features1, 'point_features2': point_features2}

class PointEncoder(nn.Module):
    def __init__(self, d_model=1024, cat_points=True):
        super(PointEncoder, self).__init__()
        self.cat_points = cat_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        if self.cat_points:
            self.conv3 = torch.nn.Conv1d(128, d_model-3, 1)
            self.bn3 = nn.BatchNorm1d(d_model-3)
        else:
            self.conv3 = torch.nn.Conv1d(128, d_model, 1)
            self.bn3 = nn.BatchNorm1d(d_model)
    def forward(self, points):
        x = F.relu(self.bn1(self.conv1(points)))
        x = F.relu(self.bn2(self.conv2(x)))
        point_features = self.bn3(self.conv3(x))
        if self.cat_points:
            x = torch.cat([point_features, points], dim=1)
        else:
            x = point_features
        x = x.permute(0, 2, 1)
        return x, point_features


def sort_points(correformer, x):
    b, t, n, k = x.shape
    x = x.cuda()
    sorted_seq = x[:, [0], :, :]
    sorted_frame = x[:, 0, :, :]
    corr_pred = torch.arange(n)[None, None, :].cuda().repeat([b, 1, 1])
    for frame_idx in range(t-1):
        p1 = sorted_frame
        p2 = x[:, frame_idx+1, :, :]
        corre_out_dict = correformer(p1, p2)
        corr_idx12, corr_idx21 = corre_out_dict['corr_idx12'], corre_out_dict['corr_idx21']
        sorted_frame = torch.gather(p2, 1, corr_idx12.unsqueeze(-1).repeat([1, 1, 3]))
        sorted_seq = torch.concat([sorted_seq, sorted_frame.unsqueeze(1)], dim=1)
        corr_pred = torch.concat([corr_pred, corr_idx21.unsqueeze(1)], axis=1)
    return sorted_seq, corr_pred


def get_correformer(correformer_path):
    # load a correformer model from path

    params_file_path = os.path.join(os.path.split(correformer_path)[0], 'params.pth')
    if os.path.exists(params_file_path):
        args = torch.load(params_file_path)
        correformer_type = args.transformer_type
        correformer_dims = args.dim
        correformer_nheads = args.n_heads
        correformer_feedforward = args.d_feedforward
        cat_points = args.cat_points
    else:
        #support for old logs that did not save params file - delete for publication
        params_str = correformer_path.split("/")[-2].split("_")[2]
        correformer_dims = int(params_str[params_str.index('d') + 1:params_str.index('h')])
        correformer_nheads = int(params_str[params_str.index('h') + 1:])
        correformer_feedforward = int(params_str[params_str.index('ff') + 2:params_str.index('d')])
        correformer_type = 'none'
        cat_points = False
    correformer = CorreFormer(d_model=correformer_dims, nhead=correformer_nheads, num_encoder_layers=6,
                                   num_decoder_layers=1, dim_feedforward=correformer_feedforward,
                              transformer_type=correformer_type, cat_points=cat_points).cuda()
    correformer.load_state_dict(torch.load(correformer_path)["model_state_dict"])
    correformer.eval()
    # correformer.train(False)
    return correformer



