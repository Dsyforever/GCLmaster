import torch
import random
from torchvision import transforms
from torch import nn

class noise(nn.Module):
    def __init__(self,Q):
        super().__init__()
        self.Q=Q

    def forward(self, xb):
        Aug_xb = xb + self.Q * 2 * (torch.randn(xb.shape) - 0.5).to(xb.device)
        return Aug_xb


def random_tran(func, p, x):
    """Randomly apply function func to x with probability p."""
    if random.random() >= p:
        return x
    else:
        return func(x)


def torch_resize(image, h, w):
    """Resize images to specific size
      Args:
        image: tensor of image(B,C,H,W).
        h: The height of image after resizing.
        w: The width of image after resizing.
      Returns:
        the image after resizing
    """
    resize = transforms.Resize([h, w])
    im_resize = resize(image)
    return im_resize


def torch_Crop(image, type, size, padding=None, scale=(0.08, 1.0), ratio=(0.75, 1.33)):
    """image cropping
        Args:
            image: tensor of image(B,C,H,W).
            type: the method to crop image.
            size: expected output size of the crop.
            scale: Specifies the lower and upper bounds for the random area of the crop.
            ratio: lower and upper bounds for the random aspect ratio of the crop, before resizing.
          Returns:
            the image after cropping
        """
    if type == 'CenterCrop':
        crop = transforms.CenterCrop(size)
    elif type == 'RandomCrop':
        crop = transforms.RandomCrop(size, padding=padding)
    elif type == 'RandomResizedCrop':
        crop = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio)
    return crop(image)


def torch_Flip(image, type, p=0.5, degrees=90):
    """image flip
      Args:
          image: tensor of image(B,C,H,W).
          type: the method to flip image.
          p: probability of the image being flipped
          degrees: Range of degrees to select from
    """

    if type == 'RandomHorizontalFlip':
        # Horizontally(水平地) flip the given image randomly with a given probability.
        filp = transforms.RandomHorizontalFlip(p)
    elif type == 'RandomVerticalFlip':
        # Vertically(竖直地) flip the given image randomly with a given probability.
        filp = transforms.RandomVerticalFlip(p)
    elif type == 'RandomRotation':
        # Rotate the image by angle.
        filp = transforms.RandomRotation(degrees=degrees)
    return filp(image)


def torch_ColorJitter(image, brightness, contrast, saturation, hue):
    """Randomly change the brightness, contrast, saturation and hue of an image.
      Args:
          image: tensor of image(B,C,H,W).
          brightness: How much to jitter brightness.
          contrast:
    
    """
    colorjitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
    return colorjitter(image)


def torch_Grayscale(image, p=0.1):
    """Randomly convert image to grayscale with a probability of p (default 0.1)

    """
    gray = transforms.RandomGrayscale(p=p)
    return gray(image)




