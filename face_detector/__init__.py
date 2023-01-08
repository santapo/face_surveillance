from typing import List

import numpy as np

from .retinaface import RetinaFace

FACE_DETECTOR = {
    "retinaface": RetinaFace
}


def get_face_detector(detector_name: str = "retinaface"):
    face_detector = FACE_DETECTOR.get(detector_name)
    return face_detector


class AbstractFaceDetector:
    def __init__(self):
        pass

    def detect_single_image(self, image: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError