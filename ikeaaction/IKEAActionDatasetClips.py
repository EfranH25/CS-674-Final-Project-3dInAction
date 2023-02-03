import os

import torch.utils.data
from torch.utils.data import Dataset
import pickle



class IKEAActionDatasetClips(Dataset):
    """
    IKEA Action Dataset class with pre-saved clips into pickles
    """

    def __init__(self, dataset_path, set):
        self.dataset_path = dataset_path
        self.set = set
        self.files_path = os.path.join(dataset_path, set)
        self.file_list = self.absolute_file_paths(self.files_path)
        self.file_list.sort()
        print("{}set contains {} clips".format(set, len(self.file_list)))

    def absolute_file_paths(self, directory):
        path = os.path.abspath(directory)
        return [entry.path for entry in os.scandir(path) if entry.is_file()]

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        with open(self.file_list[index], 'rb') as f:
            data = pickle.load(f)
        # data, labels, vid_idx, frame_pad = self.seq_cache.get(index)

        return data['inputs'], data['labels'], data['vid_idx'], data['frame_pad']


if __name__ == '__main__':
    dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_clips_32/'
    set = 'train'
    dataset = IKEAActionDatasetClips(dataset_path, set)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for batch_ind, data in enumerate(dataloader):
        print('')