# Author: Yizhak Ben-Shabat (Itzik), 2020
# evaluate action recognition performance using acc and mAp

import os
import argparse
import numpy as np
import sys
import torch
from sklearn.metrics import confusion_matrix
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add utils dir
import utils
import eval_utils
from DfaustDataset import DfaustActionClipsDataset as Dataset
import matplotlib.pyplot as plt
sys.path.append('evaluation')
from eval_detection import ANETdetection
from eval_classification import ANETclassification
import sklearn

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str,
                    default='../log/pn1_f32_p1024_shuffle_once/results/',
                    help='label prediction file')
parser.add_argument('--dataset_path', type=str, default='/home/sitzikbs/Datasets/dfaust/',
                    help='path to ground truth action segments json file')
parser.add_argument('--set', type=str, default='test', help='test | train set to evaluate')
parser.add_argument('--gt_segments_json_filename', type=str, default='gt_segments',
                    help='name of gt json filename to load')
args = parser.parse_args()

# load the gt and predicted data
gt_json_path = os.path.join(args.dataset_path, args.gt_segments_json_filename)
dataset = Dataset(args.dataset_path, set=args.set)
gt_labels = dataset.action_dataset.label_per_frame
results_json = os.path.join(args.results_path, args.set + '_action_segments.json')
results_npy = os.path.join(args.results_path, args.set + '_pred.npy')
pred_labels = dataset.get_actions_labels_from_json(results_json, mode='pred')

# load the predicted data
pred_data = np.load(results_npy, allow_pickle=True).item()
pred_labels = pred_data['pred_labels']
logits = pred_data['logits']

alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6,
                  0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 ])
# compute action localization mAP
anet_detection = ANETdetection(gt_json_path, results_json,
                               subset='testing', tiou_thresholds=alpha,
                               verbose=True, check_status=True)
anet_detection.evaluate()

localization_score_str = "Action localization scores: \n" \
                         "Average mAP= {} \n".format(anet_detection.average_mAP) + \
                         "alpha = " + " & ".join(str(alpha).split()) + "\n " \
                         "mAP scores =" +\
                         " & ".join(str(np.around(anet_detection.mAP, 2)).split()) + "\n "

# Compute classification mAP
anet_classification = ANETclassification(gt_json_path, results_json, subset='testing', verbose=True)
anet_classification.evaluate()
classification_score_str = "Action classification scores: \n" \
                            "mAP = {} \n".format(anet_classification.ap.mean())

# Compute accuracy
acc1_per_vid = []
acc3_per_vid = []

gt_single_labels = []
for vid_idx in range(len(logits)):
    single_label_per_frame = torch.tensor(gt_labels[vid_idx])
    acc1, acc3 = eval_utils.accuracy(torch.tensor(logits[vid_idx]), single_label_per_frame, topk=(1, 3))
    acc1_per_vid.append(acc1.item())
    acc3_per_vid.append(acc3.item())
    gt_single_labels.append(single_label_per_frame)

scores_str = 'top1, top3 accuracy: {} & {}\n'.format(round(np.mean(acc1_per_vid), 2), round(np.mean(acc3_per_vid), 2))

print(scores_str)
balanced_acc_per_vid = []

for vid_idx in range(len(logits)):
    single_label_per_frame = torch.tensor(gt_labels[vid_idx])
    acc = sklearn.metrics.balanced_accuracy_score(single_label_per_frame, np.argmax(logits[vid_idx], 1),
                                                  sample_weight=None, adjusted=False)
    balanced_acc_per_vid.append(acc)

balanced_score_str = 'balanced accuracy: {}'.format(round(np.mean(balanced_acc_per_vid)*100, 2))
print(balanced_score_str)


# output the dataset total score
with open(os.path.join(args.results_path, 'scores.txt'), 'w') as file:
    file.writelines(localization_score_str)
    file.writelines(classification_score_str)
    file.writelines(scores_str)
    file.writelines(balanced_score_str)

# Compute confusion matrix
print('Comptuing confusion matrix...')

c_matrix = confusion_matrix(np.concatenate(gt_single_labels).ravel(), np.concatenate(pred_labels).ravel(),
                            labels=range(dataset.action_dataset.num_classes))
class_names = utils.squeeze_class_names(dataset.action_dataset.actions)

fig, ax = utils.plot_confusion_matrix(cm=c_matrix,
                      target_names=class_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=True)

plt.savefig(os.path.join(args.results_path, 'confusion_matrix.png'))

