import cv2
from PIL import Image
import numpy as np
import os
from copy import deepcopy
from torchvision import transforms as T

def to_tensor(img):
    """
        Convert an image to a PyTorch tensor.
    """
    return T.ToTensor()(img)

def normalize(img, config):
    """
    Normalize an image.
    """
    mean = config['mean']
    std = config['std']
    normalize = T.Normalize(mean=mean, std=std)
    return normalize(img)

def load_rgb(file_path, config):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = config['resolution'] # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif self.lmdb:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

def load(image, config):
    # Load the image
    size = 256
    #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    #image = load_rgb(image_path, config)
    image_tensors = []
    landmark_tensors = []
    mask_tensors = []

    mask = None
    landmarks = None

    image = np.array(image)  # Convert to numpy array for data augmentation
    image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
    image_trans = normalize(to_tensor(image_trans), config)

    image_tensors.append(image_trans)
    landmark_tensors.append(landmarks_trans)
    mask_tensors.append(mask_trans)

    return image_tensors, landmark_tensors, mask_tensors

