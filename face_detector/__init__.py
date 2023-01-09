from .retinaface import RetinaFace

FACE_DETECTOR = {
    "retinaface": RetinaFace
}


def get_face_detector(detector_name: str = "retinaface"):
    face_detector = FACE_DETECTOR.get(detector_name)
    return face_detector