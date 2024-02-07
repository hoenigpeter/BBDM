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
        #image_aug = self.seq_syn(image=np.array(image))  # Make sure image is a numpy array

        # Convert back to PIL Image if necessary, or to torch.Tensor
        # image_aug = PIL.Image.fromarray(image_aug)
        # image_aug = transforms.ToTensor()(image_aug)

        return image, label

# Define your augmentations
seq_syn = iaa.Sequential([
    # Sometimes(0.5, PerspectiveTransform(0.05)),
    # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
    # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
    iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
    iaa.Sometimes(0.4, iaa.GaussianBlur((0., 3.))),
    iaa.Sometimes(0.3, iaa.pillike.EnhanceSharpness(factor=(0., 50.))),
    iaa.Sometimes(0.3, iaa.pillike.EnhanceContrast(factor=(0.2, 50.))),
    iaa.Sometimes(0.5, iaa.pillike.EnhanceBrightness(factor=(0.1, 6.))),
    iaa.Sometimes(0.3, iaa.pillike.EnhanceColor(factor=(0., 20.))),
    iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
    iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
    iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
    iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
    iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
    iaa.Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
    iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.0, 1.0)))
], random_order=True)

# Create an instance of your custom dataset
custom_dataset = CustomDataset(train_dataset, seq_syn)

# Create a DataLoader with the custom dataset
train_loader = DataLoader(custom_dataset,
                          batch_size=self.config.data.train.batch_size,
                          num_workers=8,
                          drop_last=True,
                          sampler=train_sampler)
