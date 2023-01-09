from typing import List

import numpy as np


class AbstractFaceDetector:
    def __init__(self):
        pass

    def detect_single_image(self, image: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError