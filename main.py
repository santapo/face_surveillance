import cv2
import numpy as np

from face_detector import get_face_detector
from face_recognizer import get_face_recognizer
from tracker import get_tracker


class FaceSurveillanceCore:
    def __init__(
        self,
        face_detector_name: str = "retinaface",
        face_recognizer_name: str = "deepface",
        tracker_name: str = "sort"
    ):
        self.face_detector = get_face_detector(face_detector_name)()
        self.face_recognizer = get_face_recognizer(face_recognizer_name)()
        self.mot_tracker = get_tracker(tracker_name)()

        self.trackid_to_name = {}

    def stream(self, source):
        ...

    def process_single_video(self, source: str):
        """Process a single video and output a new video"""
        cap = cv2.VideoCapture(source)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

        while(True):
            ret, ori_image = cap.read()
            if ori_image is None:
                break
            image = ori_image.copy()
            face_bboxs = self.face_detector.detect_single_image(image)
            faces = self.face_detector.extract_face(image, face_bboxs)

            face_resp = self.face_recognizer.find(faces)
            identity_names = self.face_recognizer.get_identity_names(face_resp)

            face_bboxs = np.array(face_bboxs) if len(face_bboxs) != 0 else np.empty((0, 5))
            trackers = self.mot_tracker.update(face_bboxs)

            for name, tracker in zip(identity_names, trackers):
                tracker = [int(i) for i in tracker]
                x1, y1, x2, y2, trackid = tracker
                cv2.rectangle(ori_image, (x1, y1), (x2, y2), (0,0,255), 5) #draw rectangle to main image

                # import ipdb; ipdb.set_trace()
                if not self.trackid_to_name.get(trackid) and name != "unkown":
                    self.trackid_to_name[trackid] = name

                if name == "unkown" and self.trackid_to_name.get(trackid):
                    name = self.trackid_to_name[trackid]

                text = f"{name}_{trackid}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 4)
                text_w, text_h = text_size
                cv2.rectangle(ori_image, (x1, y1), (x1+text_w, y1-text_h), (0,0,255), -1)
                cv2.putText(ori_image, text, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 4)

            out.write(ori_image)

        cap.release()
        out.release()