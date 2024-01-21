import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa

class CustomDataset(Dataset):
    def __init__(self, dataset, seq_syn):
        self.dataset = dataset
        self.seq_syn = seq_syn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # Assuming data is a tuple of (image, label)
        image, label = data

        # Apply augmentations
        image_aug = self.seq_syn(image=np.array(image))  # Make sure image is a numpy array

        # Convert back to PIL Image if necessary, or to torch.Tensor
        # image_aug = PIL.Image.fromarray(image_aug)
        # image_aug = transforms.ToTensor()(image_aug)

        return image_aug, label

# Define your augmentations
prob = 1.0
seq_syn = iaa.Sequential([
    iaa.Sometimes(0.5 * prob, iaa.CoarseDropout(p=0.2, size_percent=0.05)),
    iaa.Sometimes(0.5 * prob, iaa.GaussianBlur(1.2 * np.random.rand())),
    iaa.Sometimes(0.5 * prob, iaa.Add((-25, 25), per_channel=0.3)),
    iaa.Sometimes(0.3 * prob, iaa.Invert(0.2, per_channel=True)),
    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4))),
    iaa.Sometimes(0.5 * prob, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
], random_order=False)

# Create an instance of your custom dataset
custom_dataset = CustomDataset(train_dataset, seq_syn)

# Create a DataLoader with the custom dataset
train_loader = DataLoader(custom_dataset,
                          batch_size=self.config.data.train.batch_size,
                          num_workers=8,
                          drop_last=True,
                          sampler=train_sampler)
