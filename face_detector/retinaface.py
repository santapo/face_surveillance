from typing import List

import numpy as np
import tensorflow as tf
from retinaface.commons import postprocess, preprocess
from retinaface.model import retinaface_model

from .base_detector import AbstractFaceDetector

nms_threshold = 0.4; decay4 = 0.5
_feat_stride_fpn = [32, 16, 8]
_anchors_fpn = {
    'stride32': np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
    'stride16': np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
    'stride8': np.array([[-8., -8., 23., 23.], [ 0.,  0., 15., 15.]], dtype=np.float32)
}
_num_anchors = {'stride32': 2, 'stride16': 2, 'stride8': 2}

class RetinaFace(AbstractFaceDetector):
    def __init__(self):
        super().__init__()
        self._build_model()

    def _build_model(self):
        self.model = tf.function(
            retinaface_model.build_model(),
            input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),))

    def detect_single_image(
        self,
        image: np.ndarray,
        threshold: float =0.9,
        allow_upscaling: bool = True,
    ) -> List[np.ndarray]:

        proposals_list = []
        scores_list = []
        landmarks_list = []
        im_tensor, im_info, im_scale = preprocess.preprocess_image(image, allow_upscaling)
        net_out = self.model(im_tensor)
        net_out = [elt.numpy() for elt in net_out]
        sym_idx = 0

        for _idx, s in enumerate(_feat_stride_fpn):
            _key = 'stride%s'%s
            scores = net_out[sym_idx]
            scores = scores[:, :, :, _num_anchors['stride%s'%s]:]

            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

            A = _num_anchors['stride%s'%s]
            K = height * width
            anchors_fpn = _anchors_fpn['stride%s'%s]
            anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.reshape((-1, 1))

            bbox_stds = [1.0, 1.0, 1.0, 1.0]
            bbox_deltas = bbox_deltas
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * bbox_stds[3]
            proposals = postprocess.bbox_pred(anchors, bbox_deltas)

            proposals = postprocess.clip_boxes(proposals, im_info[:2])

            if s==4 and decay4<1.0:
                scores *= decay4

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0:4] /= im_scale
            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_deltas = net_out[sym_idx + 2]
            landmark_pred_len = landmark_deltas.shape[3]//A
            landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
            landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            landmarks[:, :, 0:2] /= im_scale
            landmarks_list.append(landmarks)
            sym_idx += 3

        proposals = np.vstack(proposals_list)
        if proposals.shape[0]==0:
            landmarks = np.zeros( (0,5,2) )
            return np.zeros( (0,5) ), landmarks
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

        #nms = cpu_nms_wrapper(nms_threshold)
        #keep = nms(pre_det)
        keep = postprocess.cpu_nms(pre_det, nms_threshold)

        det = np.hstack( (pre_det, proposals[:,4:]) )
        det = det[keep, :]
        landmarks = landmarks[keep]

        resp = {}
        for idx, face in enumerate(det):

            label = 'face_'+str(idx+1)
            resp[label] = {}
            resp[label]["score"] = face[4]

            resp[label]["facial_area"] = list(face[0:4].astype(int))

            resp[label]["landmarks"] = {}
            resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
            resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
            resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
            resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
            resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])

        face_bboxs = [v["facial_area"] + [v["score"]] for k, v in resp.items()]
        return face_bboxs

    def extract_face(
        self,
        image: np.ndarray,
        face_bboxs: List[np.ndarray],
    ) -> List[np.ndarray]:
        resp = []
        for bbox in face_bboxs:
            facial_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            resp.append(facial_image[:, :, ::-1])
        return resp
