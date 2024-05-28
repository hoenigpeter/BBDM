from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa
import numpy as np
import torch

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name

class ImagePathDataset_Augmented(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

        # prob = 1.0
        # self.seq_syn = iaa.Sequential([
        #     iaa.Sometimes(0.5 * prob, iaa.CoarseDropout(p=0.2, size_percent=0.05)),
        #     iaa.Sometimes(0.5 * prob, iaa.GaussianBlur(1.2 * np.random.rand())),
        #     iaa.Sometimes(0.5 * prob, iaa.Add((-25, 25), per_channel=0.3)),
        #     iaa.Sometimes(0.3 * prob, iaa.Invert(0.2, per_channel=True)),
        #     iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        #     iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4))),
        #     iaa.Sometimes(0.5 * prob, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
        # ], random_order=False)

        self.seq_syn = iaa.Sequential([
            # Sometimes(0.5, PerspectiveTransform(0.05)),
            # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
            # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
            iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
            iaa.Sometimes(0.4, iaa.GaussianBlur((0., 3.))),
            # iaa.Sometimes(0.3, iaa.pillike.EnhanceSharpness(factor=(0., 50.))),
            # iaa.Sometimes(0.3, iaa.pillike.EnhanceContrast(factor=(0.2, 50.))),
            # iaa.Sometimes(0.5, iaa.pillike.EnhanceBrightness(factor=(0.1, 6.))),
            # iaa.Sometimes(0.3, iaa.pillike.EnhanceColor(factor=(0., 20.))),
            iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
            iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
            # iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
            iaa.Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
            # iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.0, 1.0)))
        ], random_order=True)


    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def torch_to_numpy(self, tensor):
        """ Convert a torch tensor to a numpy array. """
        # Move tensor to CPU and convert to numpy
        numpy_img = tensor.cpu().numpy()

        # Change range from [-1, 1] to [0, 1] and reshape
        numpy_img = (numpy_img + 1) / 2.0
        numpy_img = numpy_img.transpose(1,2,0)

        # Convert to uint8
        numpy_img = (numpy_img * 255).astype(np.uint8)
        return numpy_img  # Remove batch dimension

    def numpy_to_torch(self, numpy_img):
        """ Convert a numpy array back to a torch tensor. """
        # Convert back to float and normalize to [-1, 1]
        torch_img = ((numpy_img / 255.0) - 0.5) * 2.0

        torch_img = torch_img.transpose(2, 0, 1)

        # Convert to torch tensor
        torch_img = torch.from_numpy(torch_img).float()

        return torch_img

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        # numpy_img = self.torch_to_numpy(image)
        # augmented_img = self.seq_syn.augment_image(numpy_img)
        # image = self.numpy_to_torch(augmented_img)

        image_name = Path(img_path).stem
        return image, image_name

