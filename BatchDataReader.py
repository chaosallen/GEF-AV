import torch
import os
import natsort
import cv2
from torch.utils.data.dataset import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Error fetching version info*")

class CubeDataset(Dataset):
    def __init__(self, data_dir, data_size, input_filename, label_filename, energy_filename,is_dataaug=False):
        self.is_dataaug = is_dataaug
        self.input_filename = input_filename
        self.label_filename = label_filename
        self.energy_filename = energy_filename
        self.data_size = data_size
        self.datasetlist = {'data': {}, 'label': {}, 'energy': {}}

        for modal in input_filename:
            self.datasetlist['data'][modal] = {}
            imglist = natsort.natsorted(os.listdir(os.path.join(data_dir, modal)))
            for img in imglist:
                self.datasetlist['data'][modal][img] = os.path.join(data_dir, modal, img)

        for modal in label_filename:
            self.datasetlist['label'][modal] = {}
            imglist = natsort.natsorted(os.listdir(os.path.join(data_dir, modal)))
            for img in imglist:
                self.datasetlist['label'][modal][img] = os.path.join(data_dir, modal, img)

        for modal in energy_filename:
            self.datasetlist['energy'][modal] = {}
            imglist = natsort.natsorted(os.listdir(os.path.join(data_dir, modal)))
            for img in imglist:
                self.datasetlist['energy'][modal][img] = os.path.join(data_dir, modal, img)


    def __getitem__(self, index):

        data_channels = len(self.input_filename)
        data = np.zeros((data_channels, self.data_size[0], self.data_size[1]), dtype=np.float32)
        label = np.zeros((self.data_size[0], self.data_size[1], len(self.label_filename)), dtype=np.float32)
        energy = np.zeros((self.data_size[0], self.data_size[1], len(self.energy_filename)), dtype=np.float32)

        # ---- Read images ----
        for i, modal in enumerate(self.input_filename):
            name = list(self.datasetlist['data'][modal])[index]
            image = cv2.imread(self.datasetlist['data'][modal][name], cv2.IMREAD_GRAYSCALE)
            image= cv2.resize(image, (self.data_size[1], self.data_size[0]),
            interpolation = cv2.INTER_LINEAR)
            image=image.astype(np.float32)
            data[i,:, :] = image / 255.0

        for j, modal in enumerate(self.label_filename):
            name = list(self.datasetlist['label'][modal])[index]
            lbl = cv2.imread(self.datasetlist['label'][modal][name], cv2.IMREAD_GRAYSCALE)
            lbl = cv2.resize(lbl, (self.data_size[1], self.data_size[0]),
            interpolation = cv2.INTER_LINEAR)
            _, lbl = cv2.threshold(lbl, 127, 255, cv2.THRESH_BINARY)
            lbl=lbl.astype(np.float32)
            label[:, :, j] = lbl / 255.0

        for k, modal in enumerate(self.energy_filename):
            name = list(self.datasetlist['energy'][modal])[index]
            ebl = cv2.imread(self.datasetlist['energy'][modal][name], cv2.IMREAD_GRAYSCALE)
            ebl = cv2.resize(ebl, (self.data_size[1], self.data_size[0]),
                               interpolation=cv2.INTER_LINEAR)
            ebl=ebl.astype(np.float32)
            energy[:, :, k] = ebl / 255.0


        if self.is_dataaug:
            data, label, energy = self.augmentation(data, label, energy)

        # Convert to tensor
        data = torch.from_numpy(data.copy())
        label = torch.from_numpy(label.copy())
        energy = torch.from_numpy(energy.copy())

        return data, label, energy, name

    def __len__(self):
        return len(self.datasetlist['data'][self.input_filename[0]])

    def augmentation(self, image, label, energy):
        """
        Perform corresponding augmentation on images, labels, and energy maps:
        - Geometric transformations are applied synchronously to image, label, and energy map
        - Lighting and blur augmentations are applied only to the image
        Inputs and outputs:
          image: numpy array, shape (C, H, W)
          label: numpy array, shape (H, W, C_label)
          energy: numpy array, shape (H, W, C_energy)
        Returns augmented versions of the three with the same shapes (float32)
        """
        seed = random.randint(0, 10000)
        # Convert to HWC format for albumentations
        image_hwc =np.transpose(image, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        label_hwc = label  # (H,W,C_label)
        energy_hwc = energy  # (H,W,C_energy)

        # Define unified geometric augmentation, treat label and energy as images
        geometric_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
        ], additional_targets={
            'label': 'image',
            'energy': 'image'
        })

        # Ensure reproducibility of random numbers (albumentations uses np.random internally)
        random.seed(seed)
        np.random.seed(seed)

        augmented = geometric_transform(image=image_hwc, label=label_hwc, energy=energy_hwc)
        image_hwc = augmented['image']
        label_hwc = augmented['label']
        energy_hwc = augmented['energy']

        # Apply lighting and blur augmentation only to the image
        lighting_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
            A.ElasticTransform(alpha=0.5, sigma=20, p=0.1),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=0.2, p=0.1),
        ])

        image_hwc = lighting_transform(image=image_hwc)['image']

        # Convert back to CHW
        image_aug = np.transpose(image_hwc, (2, 0, 1))

        # Clip and round to prevent value drift
        label_aug = np.round(label_hwc.clip(0, 1))
        energy_aug = energy_hwc.clip(0, 1)

        return image_aug.astype(np.float32), label_aug.astype(np.float32), energy_aug.astype(np.float32)
