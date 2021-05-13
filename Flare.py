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
        img = np.array(img) #PIL image를 numpy array로 변환
        h, w, _ = img.shape #image의 height, width 추출

        overlay = img.copy() #image와 합칠 overlay 레이어 선언

        max_r = min(h,w)//6 #원의 최대 반지름을 제한
        gradient = np.random.randint(h)/np.random.randint(1,w) #top-left corner와 random pixel의 gradient
        alpha = 0.3 #overlay의 투명도, 즉 원들의 투명도

        for _ in range(self.n_circles): #선언한 flare circle의 수 만큼 반복
          x = np.random.randint(w) #random한 x좌표 선택
          center = (int(gradient*x), x) #원의 중심 좌표 설정

          c = np.random.randint(190,255) #원의 색(grey) 설정
          cv2.circle(overlay, center, np.random.randint(max_r), (c,c,c), -1) #overlay 레이어 위에 원을 그림

        img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0) #overlay에 투명도 부여 후 원본 image와 합성
        img = Image.fromarray(img) #numpy array를 PIL image로 변환

        return img