from face_detector import get_face_detector
from face_recognizer import get_face_recognizer
from tracker import get_tracker



class FaceSurveillanceCore:
    def __init__(self, face_detector_name, face_recognizer_name, tracker_name):
        self.face_detector = get_face_detector(face_detector_name)
        self.face_recognizer = get_face_recognizer(face_recognizer_name)
        self.tracker = get_tracker(tracker_name)

    def stream(self, source):
        ...