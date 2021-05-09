import cv2
import numpy as np
from PIL import Image

class Flare(object):
    """Randomly creates flare on an image.

    Args:
        n_circles (int): Number of lens flare circles on each image.
    """
    def __init__(self, n_circles):
        self.n_circles = n_circles

    def __call__(self, img):
        """
        Args:
            img (PIL): PIL image.
        Returns:
            PIL: Image with randomly generated lens flare effect.
        """
        img = np.array(img)
        h, w, _ = img.shape

        overlay = img.copy()

        max_r = min(h,w)//6
        gradient = np.random.randint(h)/np.random.randint(1,w)
        alpha = 0.3

        for _ in range(self.n_circles):
          x = np.random.randint(w)
          center = (int(gradient*x), x)

          c = np.random.randint(190,255)
          cv2.circle(overlay, center, np.random.randint(max_r), (c,c,c), -1)

        img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        img = Image.fromarray(img)

        return img