import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import PIL.Image as Image


class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label, portion=0.1, mode="train", device=torch.device("cuda"), dataname="mnist", mark_dir=None, alpha = 1.0, train=False):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.train = train
        self.alpha = alpha
        if dataname == 'cifar10':
            if train:
                self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
        self.data, self.targets = self.add_trigger(self.reshape(dataset.data, dataname), dataset.targets, trigger_label, portion, mode, mark_dir)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1 # 把num型的label变成10维列表。
        label = torch.Tensor(label) 

        img = img.to(self.device)
        if self.dataname == 'cifar10':
            img = self.transform(img)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data, dataname="mnist"):
        if dataname == "mnist":
            new_data = data.reshape(len(data),1,28,28)
        elif dataname == "cifar10":
            new_data = data.reshape(len(data),32,32,3)
            new_data = data.transpose(0,3,1,2)
        return np.array(new_data)

    def norm(self, data):
        offset = np.mean(data, 0)
        scale  = np.std(data, 0).clip(min=1)
        return (data- offset) / scale

    def add_trigger(self, data, targets, trigger_label, portion, mode, mark_dir):
        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_data = new_data / 255.0 # cast from [0, 255] to [0, 1]
        new_targets = copy.deepcopy(targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        if mark_dir is None:
            """
            A white square at the right bottom corner
            """
            for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
                new_targets[idx] = trigger_label
                for c in range(channels):
                    new_data[idx, c, width-3, height-3] = 1
                    new_data[idx, c, width-3, height-2] = 1
                    new_data[idx, c, width-2, height-3] = 1
                    new_data[idx, c, width-2, height-2] = 1
            new_data = torch.Tensor(new_data)
        else:
            """
            User specifies the mark's path, plant it into inputs.
            """
            alpha = self.alpha # transparency of the mark
            mark = Image.open(mark_dir)
            mark = mark.resize((width, height), Image.ANTIALIAS) # scale the mark to the size of inputs

            if channels == 1:
                mark = np.array(mark)[:, :, 0] / 255.0 # cast from [0, 255] to [0, 1]
            elif channels == 3:
                mark = np.array(mark).transpose(2, 0, 1) / 255.0 # cast from [0, 255] to [0, 1]
            else:
                print("Channels of inputs must be 1 or 3!")
                exit
            
            """Specify Trigger's Mask"""
            mask = torch.Tensor(1 - (mark > 0.1)) # white trigger
            # mask = torch.Tensor(1 - (mark < 0.1)) # black trigger
            # mask = torch.zeros(mark.shape) # no mask

            new_data = torch.Tensor(new_data)
            for idx in perm: # if image in perm list, add trigger into img and change the label to trigger_label
                new_targets[idx] = trigger_label
                """2 Attaching Implementation
                    - directly adding `alpha * mark` to inputs, and `mask` is useless
                    - mixing with the original input entries [i, j] where mask[i, j] == 0
                """
                # new_data[idx, :, :, :] += mark * alpha
                new_data[idx, :, :, :] = torch.mul(new_data[idx, :, :, :] * (1 - alpha) + mark * alpha, 1 - mask) + torch.mul(new_data[idx, :, :, :], mask)

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return new_data, new_targets
