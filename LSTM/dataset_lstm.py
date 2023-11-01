import torch
import json
from torch.nn.utils.rnn import pad_sequence
import random
# Define a custom dataset for loading JSON data
class JSONDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data=data

    def __getitem__(self, index):
            return torch.tensor(self.data[index]['encoding'], dtype=torch.float),torch.tensor(self.data[index]['label'], dtype=torch.float)

    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):
    tensors, targets = zip(*batch)
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.vstack(targets)
    return features, targets