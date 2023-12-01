import os
import numpy as np
import random
from PIL import Image, ImageFile, ImageFilter
import torch
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class ImageCaptionDatasetCC12M(torch.utils.data.Dataset):
    def __init__(self, root, metadata, transform=None, tokenizer=None):
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples = np.load(metadata, allow_pickle=True)

    def __getitem__(self, index):
        ann = self.samples[index]
        image_name, captions = ann['image_name'], ann['captions']
        path = os.path.join(self.root, image_name)
        img = pil_loader(path)
        caption = np.random.choice(captions)

        if self.transform is not None:
            img = self.transform(img)

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return img, caption

    def __len__(self):
        return len(self.samples)
    
    
def get_cc12m_dataset(root_dir, metadata_path):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(1., 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = ImageCaptionDatasetCC12M(root=root_dir, metadata=metadata_path, transform=transform)

    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # mean = (0, 0, 0)
    # std = (1, 1, 1)
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            scale=(0.2, 1.0),
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    size = 292
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BILINEAR if args.interpolation == 'bilinear' else
                          PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
