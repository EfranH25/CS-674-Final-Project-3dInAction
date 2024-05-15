import torch
import pickle
import numpy as NumPy
import torch.cuda as tcuda
import os

# If CUDA is available, run code on GPU. Else, run code on CPU.
if tcuda.is_available():
  this_device = torch.device('cuda:0')
else:
  this_device = torch.device('cpu')

class MSRAction3DDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, set, cfg_data):
    # Call the __init__ function of the Dataset superclass.
    super(MSRAction3DDataset, self).__init__()

    with open(os.path.join(dataset_path, 'MSRAction3D_FPS_Videos.pickle') , mode = 'rb') as msr_action3d_pickle_file:
      msr_action3d_depth_map_sequences = pickle.load(msr_action3d_pickle_file)
    msr_action3d_depth_map_sequence_labels = NumPy.load(os.path.join(dataset_path, 'MSRAction3D_FPS_Video_Labels.npz'))['labels']

    # Store the number of depth maps per clip in a variable.
    clip_size = 8

    # Initialize a dictionary to hold the MSRAction3D clips and their labels.
    MSRAction3D_dataset = {'labels': [], 'clips': []}

    # Loop through the depth map sequences of the MSRAction3D Dataset.
    for index, depth_map_sequence in enumerate(msr_action3d_depth_map_sequences):
      label = msr_action3d_depth_map_sequence_labels[index]
      num_depth_maps = depth_map_sequence.shape[1]
      clips = list(torch.Tensor(depth_map_sequence[0]).split(clip_size))
      if num_depth_maps % clip_size != 0:
        clips = clips[:-1]
      MSRAction3D_dataset['labels'].append(label + torch.zeros((num_depth_maps // clip_size, )))
      [MSRAction3D_dataset['clips'].append(clip[None]) for clip in clips]

    # Concatenate all the label Tensors in the dataset together to form a single Tensor. Similarly, 
    # concatenate all the clip Tensors into a single Tensor.
    #labels = torch.concatenate(MSRAction3D_dataset['labels'])
    #subsequences = torch.concatenate(MSRAction3D_dataset['clips'])
    
    labels = torch.concat(MSRAction3D_dataset['labels'])
    subsequences = torch.concat(MSRAction3D_dataset['clips'])

    # Calculate the portion of the dataset to use depending on whether
    # the current phase is a training, validation or test phase.
    if set == "train":
      start_index = 0
      end_index = int(0.6 * labels.shape[0])
    elif set == "test":
      start_index = int(0.6 * labels.shape[0])
      end_index = labels.shape[0]

    # Store the labels and subsequences in class variables.
    self.labels = labels[start_index: end_index].to(device = this_device)
    self.subsequences = subsequences[start_index: end_index].to(device = this_device)

    # Store the clips in the dataset in a variable.
    self.clip_set = MSRAction3D_dataset['clips']
  def __getitem__(self, integral_key):
    # Implemented the __getitem__ method of this dataset.
    return (self.labels[integral_key], self.subsequences[integral_key])
  def __len__(self):
    # Implemented the __len__ method of this dataset.
    return len(self.labels)
  def __getitems__(self, batch_indices):
    # Implemented the __getitems__ method of this dataset.
    return (self.labels[batch_indices], self.subsequences[batch_indices])
  def make_weights_for_balanced_classes(self):
    label_counts = self.labels.unique(return_counts = True)[1]
    self.num_classes = label_counts.shape[0]
    return list(label_counts / label_counts.sum())