import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
import math
from torchvision.transforms import ToPILImage

class ExtRandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        # assert img.size == lbl.size
        # scale = random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = img.size
        rand_log_scale = math.log(self.scale_range[0], 2) + random.random() * (math.log(self.scale_range[1], 2) - math.log(self.scale_range[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        image = img.resize(new_size, Image.ANTIALIAS)
        mask = lbl.resize(new_size, Image.NEAREST)
        return image, mask

if __name__ == '__main__':
    h = 8*8
    w = 8*8
    B = 2
    I_ = 2*torch.eye(h, w).rot90()
    lbl = torch.ones(size=(h, w)) - torch.eye(h, w) + I_
    x = torch.rand(size=(h, w, 3)).float()
    scaler = ExtRandomScale([0.5,2])
    for i in range(10):
        x_s, y_s = scaler(ToPILImage()(x), ToPILImage()(lbl))
        print(x_s.size, y_s.size)

